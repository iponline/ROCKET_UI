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

START_BYTE = 0xAA
PKT_ATTITUDE = 0x10
PKT_THROTTLE = 0x30
PAYLOAD_LEN = 12

WINDOW_SEC = 10
TIMER_MS = 10
MAX_LEN = 2000

DEFAULT_BAUDRATE = 115200
BAUD_RATES = [
    9600, 19200, 38400, 57600, 115200, 230400, 250000, 500000
]
CONFIG_FILE = "settings.ini"


class SerialManager(QtCore.QObject):
    new_attitude = QtCore.pyqtSignal(float, float, float, float)  # (timestamp, pitch, roll, yaw)
    status_changed = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._thread = None
        self.ser = None

    def open(self, port, baud):
        self.close()
        try:
            self.ser = serial.Serial(port, baud, timeout=0.2)
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
                self.ser.close()
            except:
                pass
        self.ser = None
        self.status_changed.emit("Disconnected")

    def _read_thread(self):
        start_time = time.time()
        while self._running:
            try:
                payload = self.find_packet(self.ser)
                if payload:
                    pitch, roll, yaw = struct.unpack('<fff', payload)
                    now = time.time() - start_time
                    self.new_attitude.emit(now, pitch, roll, yaw)
            except Exception as e:
                self.status_changed.emit(f"Error: {e}")
                self.close()
                break

    def calc_checksum(self, buf):
        cksum = 0
        for b in buf:
            cksum ^= b
        return cksum

    def find_packet(self, ser):
        while self._running:
            b = ser.read(1)
            if not b:
                continue
            if b[0] == START_BYTE:
                header = ser.read(2)
                if len(header) < 2:
                    continue
                pkt_type, pkt_len = header
                if pkt_type != PKT_ATTITUDE or pkt_len != PAYLOAD_LEN:
                    continue
                rest = ser.read(PAYLOAD_LEN + 1)
                if len(rest) < PAYLOAD_LEN + 1:
                    continue
                payload = rest[:-1]
                checksum = rest[-1]
                full_packet = bytes([START_BYTE, pkt_type, pkt_len]) + payload
                if self.calc_checksum(full_packet) == checksum:
                    return payload

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


class PIDPlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__(title="ROCKET PID TUNING V1.0")
        self.plot = self.addPlot()
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'Angle', units='deg')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.addLegend()
        self.curve_pitch = self.plot.plot(pen=pg.mkPen('r', width=2), name='Pitch')
        self.curve_roll = self.plot.plot(pen=pg.mkPen('b', width=2), name='Roll')
        self.plot.setYRange(-90, 90)
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        self.times = deque(maxlen=MAX_LEN)
        self.pitches = deque(maxlen=MAX_LEN)
        self.rolls = deque(maxlen=MAX_LEN)

    def add_data(self, now, pitch, roll, yaw):
        self.times.append(now)
        self.pitches.append(pitch)
        self.rolls.append(roll)

    def smooth(self, arr, window=9, poly=2):
        if len(arr) >= window:
            return savgol_filter(arr, window, poly)
        return arr

    def update_plot(self):
        if len(self.times) > 2:
            t0 = self.times[0]
            t_arr = np.array(self.times) - t0
            p_arr = np.array(self.pitches)
            r_arr = np.array(self.rolls)
            mask = t_arr > (t_arr[-1] - WINDOW_SEC)
            t_disp = t_arr[mask]
            p_disp = p_arr[mask]
            r_disp = r_arr[mask]
            p_disp_smooth = self.smooth(p_disp)
            r_disp_smooth = self.smooth(r_disp)
            self.curve_pitch.setData(t_disp, p_disp_smooth)
            self.curve_roll.setData(t_disp, r_disp_smooth)
            self.plot.setXRange(max(0, t_disp[-1] - WINDOW_SEC), t_disp[-1], padding=0)

    def reset_data(self):
        self.times.clear()
        self.pitches.clear()
        self.rolls.clear()
        self.curve_pitch.clear()
        self.curve_roll.clear()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROCKET PID TUNING V1.0")
        self.setWindowIcon(QtGui.QIcon())

        # ----------- Top Controls -----------
        com_label = QtWidgets.QLabel("COM Port:")
        self.com_combo = QtWidgets.QComboBox()
        self.refresh_ports()

        baud_label = QtWidgets.QLabel("Baud:")
        self.baud_combo = QtWidgets.QComboBox()
        for b in BAUD_RATES:
            self.baud_combo.addItem(str(b))
        self.baud_combo.setCurrentText(str(DEFAULT_BAUDRATE))

        self.connect_btn = QtWidgets.QPushButton("Connect")

        # Full/ Kill Throttle button
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

        # ----------- Plot Widget ------------
        self.plot_widget = PIDPlotWidget()

        # ----------- Status Bar -------------
        self.status = QtWidgets.QLabel("Disconnected")

        # ----------- Main Layout ------------
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.status)

        # ----------- Serial Manager ---------
        self.serial_manager = SerialManager()
        self.serial_manager.new_attitude.connect(self.handle_attitude)
        self.serial_manager.status_changed.connect(self.set_status)

        # ----------- Timer ------------------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.plot_widget.update_plot)
        self.timer.start(TIMER_MS)

        # ----------- Button Connects --------
        self.connect_btn.clicked.connect(self.on_connect_toggle)
        self.throttle_btn.clicked.connect(self.on_throttle_clicked)

        # ----------- Load settings -----------
        self.load_settings()

    def closeEvent(self, event):
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
            # Currently connected → Disconnect
            self.serial_manager.close()
            self.connect_btn.setText("Connect")
            self.plot_widget.reset_data()
        else:
            # Currently disconnected → Connect
            port = self.com_combo.currentText()
            baud = int(self.baud_combo.currentText())
            if port and self.serial_manager.open(port, baud):
                self.connect_btn.setText("Disconnect")

    def handle_attitude(self, now, pitch, roll, yaw):
        self.plot_widget.add_data(now, pitch, roll, yaw)

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
