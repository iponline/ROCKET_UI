import sys
import serial
import struct
import threading
import time
from collections import deque

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import serial.tools.list_ports
import configparser
import os
from scipy.signal import savgol_filter

# ---------------- Protocol ----------------
START_BYTE = 0xAA
PKT_ATTITUDE = 0x10           # FCC -> PC: payload <fff> = pitch, roll, yaw
PKT_SETPOINTS = 0x23          # FCC -> PC: payload <ff> (also PC -> FCC request with 8 dummy bytes)
PKT_THROTTLE  = 0x30          # PC -> FCC: 0x00 kill, 0x01 full
PKT_SET_SETPOINTS = 0x24      # PC -> FCC: payload <ff> = pitch, roll

PKT_SET_GAINS = 0x20          # PC -> FCC: payload <ffffff> = pKp,pKi,pKd,rKp,rKi,rKd
PKT_GAINS     = 0x21          # FCC -> PC current gains; PC -> FCC request with 1 dummy byte

# NEW: EEPROM control
PKT_SAVE_EEPROM    = 0x50     # PC -> FCC: save current params; FCC -> PC: ACK (len=0)
PKT_RESET_DEFAULTS = 0x51     # PC -> FCC: reset to defaults & store; FCC -> PC: ACK (len=0)

PAYLOAD_LEN_ATT = 12          # 3 floats

# ---------------- UI / Plot ----------------
WINDOW_SEC = 10
TIMER_MS = 10
MAX_LEN = 2000

DEFAULT_BAUDRATE = 115200
BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400, 250000, 500000]
CONFIG_FILE = "settings.ini"


class SerialManager(QtCore.QObject):
    new_attitude = QtCore.pyqtSignal(float, float, float, float)  # (timestamp, pitch, roll, yaw)
    new_setpoints = QtCore.pyqtSignal(float, float)               # (pitch_set, roll_set)
    new_gains = QtCore.pyqtSignal(float, float, float, float, float, float)  # pKp,pKi,pKd,rKp,rKi,rKd
    status_changed = QtCore.pyqtSignal(str)

    # NEW: ACK signals
    ack_saved = QtCore.pyqtSignal()
    ack_restored = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = False
        self._thread = None
        self.ser = None

    def open(self, port, baud):
        self.close()
        try:
            self.ser = serial.Serial(port, baud, timeout=0.2)
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass

            self._running = True
            self._thread = threading.Thread(target=self._read_thread, daemon=True)
            self._thread.start()
            self.status_changed.emit("Connected")
            return True
        except Exception as e:
            self.status_changed.emit(f"Error: {e}")
            return False

    def close(self):
        self._running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass

        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

        self._thread = None
        self.ser = None
        self.status_changed.emit("Disconnected")

    def _read_thread(self):
        start_time = time.time()
        try:
            while self._running:
                try:
                    pkt = self.read_packet(self.ser)
                    if not self._running or pkt is None:
                        continue

                    pkt_type, payload = pkt

                    if pkt_type == PKT_ATTITUDE and len(payload) == PAYLOAD_LEN_ATT:
                        pitch, roll, yaw = struct.unpack('<fff', payload)
                        now = time.time() - start_time
                        self.new_attitude.emit(now, pitch, roll, yaw)

                    elif pkt_type == PKT_SETPOINTS and len(payload) >= 8:
                        sp_pitch, sp_roll = struct.unpack('<ff', payload[:8])
                        self.new_setpoints.emit(sp_pitch, sp_roll)

                    elif pkt_type == PKT_GAINS and len(payload) >= 24:
                        pKp, pKi, pKd, rKp, rKi, rKd = struct.unpack('<ffffff', payload[:24])
                        self.new_gains.emit(pKp, pKi, pKd, rKp, rKi, rKd)

                    # NEW: handle ACKs (len==0)
                    elif pkt_type == PKT_SAVE_EEPROM and len(payload) == 0:
                        self.ack_saved.emit()
                    elif pkt_type == PKT_RESET_DEFAULTS and len(payload) == 0:
                        self.ack_restored.emit()

                except Exception as e:
                    self.status_changed.emit(f"Error: {e}")
                    self._running = False
                    break
        finally:
            pass

    @staticmethod
    def calc_checksum(buf: bytes) -> int:
        cksum = 0
        for b in buf:
            cksum ^= b
        return cksum

    def read_packet(self, ser):
        while self._running and ser and ser.is_open:
            b = ser.read(1)
            if not b:
                return None
            if b[0] != START_BYTE:
                continue

            header = ser.read(2)
            if len(header) < 2:
                continue

            pkt_type, pkt_len = header
            rest = ser.read(pkt_len + 1)  # payload + checksum
            if len(rest) < pkt_len + 1:
                continue

            payload = rest[:-1]
            checksum = rest[-1]
            full_packet = bytes([START_BYTE, pkt_type, pkt_len]) + payload
            if self.calc_checksum(full_packet) == checksum:
                return (pkt_type, payload)
        return None

    def build_packet(self, pkt_type: int, payload: bytes) -> bytes:
        length = len(payload)
        header = bytes([START_BYTE, pkt_type, length])
        full_packet = header + payload
        checksum = self.calc_checksum(full_packet)
        return full_packet + bytes([checksum])

    def send_packet(self, pkt_type: int, payload: bytes = b""):
        if self.ser and self.ser.is_open:
            packet = self.build_packet(pkt_type, payload)
            self.ser.write(packet)
            print(f"Sent packet: {packet.hex()}")

    # Convenience helpers
    def send_request_setpoints(self):
        self.send_packet(PKT_SETPOINTS, b'\x00' * 8)

    def send_set_setpoints(self, pitch_deg: float, roll_deg: float):
        payload = struct.pack('<ff', float(pitch_deg), float(roll_deg))
        self.send_packet(PKT_SET_SETPOINTS, payload)

    def send_request_gains(self):
        self.send_packet(PKT_GAINS, b'\x00')

    def send_set_gains(self, pKp, pKi, pKd, rKp, rKi, rKd):
        payload = struct.pack('<ffffff',
                              float(pKp), float(pKi), float(pKd),
                              float(rKp), float(rKi), float(rKd))
        self.send_packet(PKT_SET_GAINS, payload)

    # NEW: EEPROM commands
    def send_save_to_eeprom(self):
        self.send_packet(PKT_SAVE_EEPROM, b'')

    def send_restore_defaults(self):
        self.send_packet(PKT_RESET_DEFAULTS, b'')


class PIDPlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__(title="ROCKET PID TUNING V1.0")
        self.plot = self.addPlot()
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'Angle', units='deg')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.addLegend()
        self.curve_pitch = self.plot.plot(pen=pg.mkPen('r', width=2), name='Pitch')
        self.curve_roll  = self.plot.plot(pen=pg.mkPen('b', width=2), name='Roll')
        self.plot.setYRange(-90, 90)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)  # no autorange on Y
        self.plot.getViewBox().setMouseEnabled(x=True, y=False)         # disable Y zoom/pan via mouse
        pg.setConfigOptions(antialias=True, useOpenGL=True)

        # --- X-axis behavior: fixed at startup ---
        self.window_sec = float(WINDOW_SEC)  # adjustable later
        self.auto_scroll = False             # fixed at launch
        self.plot.setXRange(0, self.window_sec, padding=0)

        self.times = deque(maxlen=MAX_LEN)
        self.pitches = deque(maxlen=MAX_LEN)
        self.rolls = deque(maxlen=MAX_LEN)

        # --- static horizontal setpoint lines + labels ---
        self.sp_pitch_value = 0.0
        self.sp_roll_value = 0.0

        self.sp_line_pitch = pg.InfiniteLine(
            pos=self.sp_pitch_value, angle=0,
            pen=pg.mkPen('r', width=1, style=QtCore.Qt.DotLine)
        )
        self.sp_line_roll = pg.InfiniteLine(
            pos=self.sp_roll_value, angle=0,
            pen=pg.mkPen('b', width=1, style=QtCore.Qt.DotLine)
        )
        self.plot.addItem(self.sp_line_pitch)
        self.plot.addItem(self.sp_line_roll)

        self.sp_text_pitch = pg.TextItem(html='<div style="color:#d00;font-weight:bold;">P:0.0</div>',
                                         anchor=(0.0, 1.0))
        self.sp_text_roll  = pg.TextItem(html='<div style="color:#08f;font-weight:bold;">R:0.0</div>',
                                         anchor=(0.0, 1.0))
        self.plot.addItem(self.sp_text_pitch)
        self.plot.addItem(self.sp_text_roll)

    def add_data(self, now, pitch, roll, yaw):
        self.times.append(now)
        self.pitches.append(pitch)
        self.rolls.append(roll)

    def smooth(self, arr, window=9, poly=2):
        if len(arr) >= window:
            return savgol_filter(arr, window, poly)
        return arr

    def set_setpoints(self, p_val: float, r_val: float):
        self.sp_pitch_value = float(p_val)
        self.sp_roll_value = float(r_val)
        self.sp_line_pitch.setValue(self.sp_pitch_value)
        self.sp_line_roll.setValue(self.sp_roll_value)
        self.sp_text_pitch.setHtml(f'<div style="color:#d00;font-weight:bold;">P:{self.sp_pitch_value:.1f}</div>')
        self.sp_text_roll.setHtml(f'<div style="color:#08f;font-weight:bold;">R:{self.sp_roll_value:.1f}</div>')
        self._position_setpoint_labels()

    def _position_setpoint_labels(self):
        try:
            (xmin, xmax), (ymin, ymax) = self.plot.getViewBox().viewRange()
        except Exception:
            return
        x = xmax - 0.02 * max(1e-6, (xmax - xmin))
        yoff = 0.04 * max(1.0, (ymax - ymin))

        yp = np.clip(self.sp_pitch_value + yoff, ymin + 1, ymax - 1)
        yr = np.clip(self.sp_roll_value  + yoff, ymin + 1, ymax - 1)

        self.sp_text_pitch.setPos(x, yp)
        self.sp_text_roll.setPos(x, yr)

    # --- controls exposed to MainWindow ---
    def set_auto_scroll(self, enabled: bool):
        self.auto_scroll = bool(enabled)
        if not self.auto_scroll:
            self.plot.setXRange(0, self.window_sec, padding=0)

    def set_window_sec(self, seconds: float):
        self.window_sec = float(max(0.1, seconds))
        if not self.auto_scroll:
            self.plot.setXRange(0, self.window_sec, padding=0)

    def update_plot(self):
        if len(self.times) > 2:
            t0 = self.times[0]
            t_arr = np.array(self.times) - t0
            p_arr = np.array(self.pitches)
            r_arr = np.array(self.rolls)

            if self.auto_scroll:
                # show last window_sec seconds and move axis
                t_end = t_arr[-1]
                t_start = max(0.0, t_end - self.window_sec)
                mask = (t_arr >= t_start) & (t_arr <= t_end)
                self.plot.setXRange(t_start, t_end, padding=0)
            else:
                # fixed axis from 0..window_sec
                mask = t_arr <= self.window_sec
                # axis already fixed in __init__/set_window_sec()

            t_disp = t_arr[mask]
            p_disp = self.smooth(p_arr[mask])
            r_disp = self.smooth(r_arr[mask])
            self.curve_pitch.setData(t_disp, p_disp)
            self.curve_roll.setData(t_disp, r_disp)
        self._position_setpoint_labels()

    def reset_data(self):
        self.times.clear()
        self.pitches.clear()
        self.rolls.clear()
        self.curve_pitch.clear()
        self.curve_roll.clear()
        if not self.auto_scroll:
            self.plot.setXRange(0, self.window_sec, padding=0)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROCKET PID TUNING V1.0")
        self.setWindowIcon(QtGui.QIcon())

        # ----------- Top Controls (COM) -----------
        com_label = QtWidgets.QLabel("COM Port:")
        self.com_combo = QtWidgets.QComboBox()
        self.refresh_ports()

        baud_label = QtWidgets.QLabel("Baud:")
        self.baud_combo = QtWidgets.QComboBox()
        for b in BAUD_RATES:
            self.baud_combo.addItem(str(b))
        self.baud_combo.setCurrentText(str(DEFAULT_BAUDRATE))

        self.connect_btn = QtWidgets.QPushButton("Connect")

        # Full / Kill Throttle button
        self.throttle_btn = QtWidgets.QPushButton("Full Throttle")
        self.throttle_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.throttle_state = False

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(com_label)
        control_layout.addWidget(self.com_combo)
        control_layout.addWidget(baud_label)
        control_layout.addWidget(self.baud_combo)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.throttle_btn)
        control_layout.addStretch()

        # ----------- Setpoints Row -----------
        latin_locale = QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)

        sp_pitch_label = QtWidgets.QLabel("Setpoint Pitch (deg):")
        self.sp_set_pitch = QtWidgets.QDoubleSpinBox()
        self.sp_set_pitch.setLocale(latin_locale)
        self.sp_set_pitch.setRange(-90.0, 90.0)
        self.sp_set_pitch.setDecimals(1)
        self.sp_set_pitch.setSingleStep(0.1)
        self.sp_set_pitch.setValue(0.0)

        sp_roll_label = QtWidgets.QLabel("Setpoint Roll (deg):")
        self.sp_set_roll = QtWidgets.QDoubleSpinBox()
        self.sp_set_roll.setLocale(latin_locale)
        self.sp_set_roll.setRange(-90.0, 90.0)
        self.sp_set_roll.setDecimals(1)
        self.sp_set_roll.setSingleStep(0.1)
        self.sp_set_roll.setValue(0.0)

        self.btn_request_setpoints = QtWidgets.QPushButton("Request Setpoints")
        self.btn_set_setpoints = QtWidgets.QPushButton("Set Setpoints")

        sp_layout = QtWidgets.QHBoxLayout()
        sp_layout.addWidget(sp_pitch_label)
        sp_layout.addWidget(self.sp_set_pitch)
        sp_layout.addSpacing(12)
        sp_layout.addWidget(sp_roll_label)
        sp_layout.addWidget(self.sp_set_roll)
        sp_layout.addSpacing(20)
        sp_layout.addWidget(self.btn_request_setpoints)
        sp_layout.addWidget(self.btn_set_setpoints)
        sp_layout.addStretch()

        # ----------- PID Gains Rows -----------
        # Pitch PID
        gp_pitch = QtWidgets.QGroupBox("Pitch PID Gains")
        gp_pitch_layout = QtWidgets.QHBoxLayout(gp_pitch)
        self.pitch_kp = QtWidgets.QDoubleSpinBox()
        self.pitch_ki = QtWidgets.QDoubleSpinBox()
        self.pitch_kd = QtWidgets.QDoubleSpinBox()
        for w in (self.pitch_kp, self.pitch_ki, self.pitch_kd):
            w.setLocale(latin_locale)
            w.setRange(0.0, 100.0)
            w.setDecimals(3)
            w.setSingleStep(0.001)
            w.setValue(0.0)
            w.setMinimumWidth(90)
        gp_pitch_layout.addWidget(QtWidgets.QLabel("Kp:")); gp_pitch_layout.addWidget(self.pitch_kp)
        gp_pitch_layout.addWidget(QtWidgets.QLabel("Ki:")); gp_pitch_layout.addWidget(self.pitch_ki)
        gp_pitch_layout.addWidget(QtWidgets.QLabel("Kd:")); gp_pitch_layout.addWidget(self.pitch_kd)
        gp_pitch_layout.addStretch()

        # Roll PID
        gp_roll = QtWidgets.QGroupBox("Roll PID Gains")
        gp_roll_layout = QtWidgets.QHBoxLayout(gp_roll)
        self.roll_kp = QtWidgets.QDoubleSpinBox()
        self.roll_ki = QtWidgets.QDoubleSpinBox()
        self.roll_kd = QtWidgets.QDoubleSpinBox()
        for w in (self.roll_kp, self.roll_ki, self.roll_kd):
            w.setLocale(latin_locale)
            w.setRange(0.0, 100.0)
            w.setDecimals(3)
            w.setSingleStep(0.001)
            w.setValue(0.0)
            w.setMinimumWidth(90)
        gp_roll_layout.addWidget(QtWidgets.QLabel("Kp:")); gp_roll_layout.addWidget(self.roll_kp)
        gp_roll_layout.addWidget(QtWidgets.QLabel("Ki:")); gp_roll_layout.addWidget(self.roll_ki)
        gp_roll_layout.addWidget(QtWidgets.QLabel("Kd:")); gp_roll_layout.addWidget(self.roll_kd)
        gp_roll_layout.addStretch()

        # Gains buttons
        gains_btn_row = QtWidgets.QHBoxLayout()
        self.btn_request_gains = QtWidgets.QPushButton("Request Gains")
        self.btn_set_gains = QtWidgets.QPushButton("Set Gains")
        gains_btn_row.addWidget(self.btn_request_gains)
        gains_btn_row.addWidget(self.btn_set_gains)
        gains_btn_row.addStretch()

        # NEW: EEPROM buttons (below PID gains, side-by-side)
        eeprom_btn_row = QtWidgets.QHBoxLayout()
        self.btn_save_eeprom = QtWidgets.QPushButton("Save to EEPROM")
        self.btn_restore_defaults = QtWidgets.QPushButton("Restore Defaults")
        eeprom_btn_row.addWidget(self.btn_save_eeprom)
        eeprom_btn_row.addWidget(self.btn_restore_defaults)
        eeprom_btn_row.addStretch()

        # ----------- X-axis controls (toggle + window seconds) -----------
        xaxis_row = QtWidgets.QHBoxLayout()
        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll X")
        self.chk_autoscroll.setChecked(False)     # start fixed
        self.chk_autoscroll.setEnabled(False)     # enable after connect
        self.win_sec_spin = QtWidgets.QDoubleSpinBox()
        self.win_sec_spin.setLocale(latin_locale)
        self.win_sec_spin.setRange(0.5, 120.0)
        self.win_sec_spin.setDecimals(1)
        self.win_sec_spin.setSingleStep(0.5)
        self.win_sec_spin.setValue(float(WINDOW_SEC))
        self.win_sec_spin.setSuffix(" s")
        self.win_sec_spin.setEnabled(False)       # enable after connect
        xaxis_row.addWidget(self.chk_autoscroll)
        xaxis_row.addWidget(QtWidgets.QLabel("Window:"))
        xaxis_row.addWidget(self.win_sec_spin)
        xaxis_row.addStretch()

        # ----------- Plot Widget -----------
        self.plot_widget = PIDPlotWidget()

        # ----------- Status Bar -----------
        self.status = QtWidgets.QLabel("Disconnected")

        # ----------- Main Layout -----------
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(sp_layout)
        main_layout.addWidget(gp_pitch)
        main_layout.addWidget(gp_roll)
        main_layout.addLayout(gains_btn_row)
        main_layout.addLayout(eeprom_btn_row)
        main_layout.addLayout(xaxis_row)          # NEW: x-axis controls
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.status)

        # ----------- Serial Manager ---------
        self.serial_manager = SerialManager()
        self.serial_manager.new_attitude.connect(self.handle_attitude)
        self.serial_manager.new_setpoints.connect(self.handle_new_setpoints)
        self.serial_manager.new_gains.connect(self.handle_new_gains)
        self.serial_manager.status_changed.connect(self.set_status)

        # NEW: ACK handlers
        self.serial_manager.ack_saved.connect(self.on_ack_saved)
        self.serial_manager.ack_restored.connect(self.on_ack_restored)

        # ----------- Timer ------------------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.plot_widget.update_plot)
        self.timer.start(TIMER_MS)

        # ----------- Buttons ----------------
        self.connect_btn.clicked.connect(self.on_connect_toggle)
        self.throttle_btn.clicked.connect(self.on_throttle_clicked)
        self.btn_request_setpoints.clicked.connect(self.on_request_setpoints_clicked)
        self.btn_set_setpoints.clicked.connect(self.on_set_setpoints_clicked)
        self.btn_request_gains.clicked.connect(self.on_request_gains_clicked)
        self.btn_set_gains.clicked.connect(self.on_set_gains_clicked)
        # NEW: EEPROM buttons
        self.btn_save_eeprom.clicked.connect(self.on_save_eeprom_clicked)
        self.btn_restore_defaults.clicked.connect(self.on_restore_defaults_clicked)
        # NEW: X-axis controls wiring
        self.chk_autoscroll.toggled.connect(self.plot_widget.set_auto_scroll)
        self.win_sec_spin.valueChanged.connect(self.plot_widget.set_window_sec)

        # ----------- Load settings ----------
        self.load_settings()

    def closeEvent(self, event):
        try:
            self.serial_manager.close()
        except Exception:
            pass
        self.save_settings()
        event.accept()

    def save_settings(self):
        config = configparser.ConfigParser()
        config['SERIAL'] = {
            'com_port': self.com_combo.currentText(),
            'baud': self.baud_combo.currentText()
        }
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

    def load_settings(self):
        self.refresh_ports()
        if os.path.exists(CONFIG_FILE):
            config = configparser.ConfigParser()
            config.read(CONFIG_FILE)
            if 'SERIAL' in config:
                baud_str = config['SERIAL'].get('baud', str(DEFAULT_BAUDRATE))
                idx = self.baud_combo.findText(baud_str)
                if idx >= 0:
                    self.baud_combo.setCurrentIndex(idx)
                com_str = config['SERIAL'].get('com_port', '')
                idx = self.com_combo.findText(com_str)
                if idx >= 0:
                    self.com_combo.setCurrentIndex(idx)

    def refresh_ports(self):
        self.com_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.com_combo.addItems(ports)
        if ports:
            self.com_combo.setCurrentIndex(0)

    def set_status(self, msg):
        self.status.setText(msg)

    def on_connect_toggle(self):
        if self.serial_manager.ser and self.serial_manager.ser.is_open:
            self.serial_manager.close()
            self.connect_btn.setText("Connect")
            self.plot_widget.reset_data()
            # disable axis controls when disconnected
            self.chk_autoscroll.setEnabled(False)
            self.win_sec_spin.setEnabled(False)
        else:
            self.plot_widget.reset_data()
            port = self.com_combo.currentText()
            baud = int(self.baud_combo.currentText())
            if port and self.serial_manager.open(port, baud):
                self.connect_btn.setText("Disconnect")
                # enable axis controls after connecting
                self.chk_autoscroll.setEnabled(True)
                self.win_sec_spin.setEnabled(True)

    def handle_attitude(self, now, pitch, roll, yaw):
        self.plot_widget.add_data(now, pitch, roll, yaw)

    def handle_new_setpoints(self, sp_pitch, sp_roll):
        self.sp_set_pitch.blockSignals(True)
        self.sp_set_roll.blockSignals(True)
        self.sp_set_pitch.setValue(float(sp_pitch))
        self.sp_set_roll.setValue(float(sp_roll))
        self.sp_set_pitch.blockSignals(False)
        self.sp_set_roll.blockSignals(False)
        self.set_status(f"Setpoints received: Pitch={sp_pitch:.1f}, Roll={sp_roll:.1f}")
        self.plot_widget.set_setpoints(sp_pitch, sp_roll)

    def handle_new_gains(self, pKp, pKi, pKd, rKp, rKi, rKd):
        for w, val in [(self.pitch_kp, pKp), (self.pitch_ki, pKi), (self.pitch_kd, pKd),
                       (self.roll_kp, rKp), (self.roll_ki, rKi), (self.roll_kd, rKd)]:
            w.blockSignals(True)
            w.setValue(float(val))
            w.blockSignals(False)
        self.set_status(f"Gains received: P(Kp={pKp:.3f},Ki={pKi:.3f},Kd={pKd:.3f}) | "
                        f"R(Kp={rKp:.3f},Ki={rKi:.3f},Kd={rKd:.3f})")

    def on_throttle_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return

        if not self.throttle_state:  # Full Throttle
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm", "Send FULL THROTTLE command?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm == QtWidgets.QMessageBox.Yes:
                self.serial_manager.send_packet(PKT_THROTTLE, bytes([0x01]))
                self.throttle_state = True
                self.throttle_btn.setText("Kill Throttle")
                self.throttle_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        else:  # Kill Throttle
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm", "Send KILL THROTTLE command?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm == QtWidgets.QMessageBox.Yes:
                self.serial_manager.send_packet(PKT_THROTTLE, bytes([0x00]))
                self.throttle_state = False
                self.throttle_btn.setText("Full Throttle")
                self.throttle_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")

    def on_request_setpoints_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return
        self.serial_manager.send_request_setpoints()
        self.set_status("Requested setpoints...")

    def on_set_setpoints_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return

        sp_p = self.sp_set_pitch.value()
        sp_r = self.sp_set_roll.value()

        confirm = QtWidgets.QMessageBox.question(
            self, "Confirm",
            f"Send SET SETPOINTS?\nPitch = {sp_p:.1f} deg, Roll = {sp_r:.1f} deg",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self.serial_manager.send_set_setpoints(sp_p, sp_r)
            self.set_status(f"Setpoints sent: Pitch={sp_p:.1f}, Roll={sp_r:.1f}")
            self.plot_widget.set_setpoints(sp_p, sp_r)

    def on_request_gains_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return
        self.serial_manager.send_request_gains()
        self.set_status("Requested gains...")

    def on_set_gains_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return

        pKp = self.pitch_kp.value(); pKi = self.pitch_ki.value(); pKd = self.pitch_kd.value()
        rKp = self.roll_kp.value();  rKi = self.roll_ki.value();  rKd = self.roll_kd.value()

        msg = (f"Send SET GAINS?\n"
               f"Pitch: Kp={pKp:.3f}, Ki={pKi:.3f}, Kd={pKd:.3f}\n"
               f"Roll:  Kp={rKp:.3f}, Ki={rKi:.3f}, Kd={rKd:.3f}")
        confirm = QtWidgets.QMessageBox.question(self, "Confirm", msg,
                                                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if confirm == QtWidgets.QMessageBox.Yes:
            self.serial_manager.send_set_gains(pKp, pKi, pKd, rKp, rKi, rKd)
            self.set_status("Gains sent.")

    # NEW: EEPROM button handlers + ACK handlers
    def on_save_eeprom_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return
        confirm = QtWidgets.QMessageBox.question(
            self, "Confirm",
            "Save current PID gains and setpoints to EEPROM?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self.serial_manager.send_save_to_eeprom()
            self.set_status("Save to EEPROM requested... (waiting for ACK)")

    def on_restore_defaults_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return
        confirm = QtWidgets.QMessageBox.question(
            self, "Confirm",
            "Restore DEFAULT parameters and write them to EEPROM?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self.serial_manager.send_restore_defaults()
            self.set_status("Restore defaults requested... (waiting for ACK)")

    def on_ack_saved(self):
        QtWidgets.QMessageBox.information(self, "EEPROM", "Saved to EEPROM (ACK).")
        self.set_status("EEPROM save ACK received.")

    def on_ack_restored(self):
        QtWidgets.QMessageBox.information(self, "EEPROM", "Defaults restored and stored (ACK).")
        self.set_status("Defaults restore ACK received.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
