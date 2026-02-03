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

# ---- Try to enable 3D rocket (OpenGL) ----
try:
    from pyqtgraph.opengl import GLViewWidget, GLMeshItem, GLAxisItem, GLGridItem, MeshData
    _HAS_GL = True
except Exception:
    _HAS_GL = False

# --- Optional STL loaders (either one is fine) ---
try:
    import trimesh  # pip install trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

try:
    from stl import mesh as stl_mesh  # pip install numpy-stl
    _HAS_NUMPY_STL = True
except Exception:
    _HAS_NUMPY_STL = False

# ---------------- Protocol ----------------
START_BYTE = 0xAA
PKT_ATTITUDE = 0x10
PKT_SETPOINTS = 0x23
PKT_THROTTLE  = 0x30
PKT_SET_SETPOINTS = 0x24
PKT_SET_GAINS = 0x20
PKT_GAINS     = 0x21
PKT_SAVE_EEPROM    = 0x50
PKT_RESET_DEFAULTS = 0x51
PKT_STICK          = 0x40  # Stick mode + throttle queries/status
PAYLOAD_LEN_ATT = 12  # 3 floats

# Query codes for PKT_STICK (by convention)
STICK_QUERY_THROTTLE = 0x03  # ask FCC for throttle status
# FCC reply is assumed to be either [status] or [0x03, status],
# where status: 0x00 = kill, 0x01 = full

# ---------------- UI / Plot ----------------
WINDOW_SEC = 10
TIMER_MS = 10
MAX_LEN = 2000

DEFAULT_BAUDRATE = 115200
BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400, 250000, 500000]
CONFIG_FILE = "settings.ini"

# STL filename (put it next to this script)
STL_FILENAME = "Falcon9.stl"

# ============ VISUAL TWEAKS ============
MODEL_OFF_WHITE = (0.98, 0.99, 0.99, 1.0)
MODEL_EDGE_COLOR = (0.50, 0.50, 0.50, 1.0)
FIN_OFF_WHITE   = (0.90, 0.90, 0.90, 1.0)
PIVOT_RAISE_FRAC = 0.12
# =======================================


def meshdata_from_stl(path: str, scale: float = 1.0) -> 'MeshData':
    if _HAS_TRIMESH:
        m = trimesh.load(path, force='mesh', skip_materials=True, process=False)
        if not hasattr(m, "vertices") or not hasattr(m, "faces"):
            raise ValueError("STL didn't load as a triangle mesh.")
        V = (np.asarray(m.vertices, dtype=np.float32) * float(scale))
        F = np.asarray(m.faces, dtype=np.int32)
        return MeshData(vertexes=V, faces=F)

    if _HAS_NUMPY_STL:
        m = stl_mesh.Mesh.from_file(path)
        tris = np.asarray(m.vectors, dtype=np.float32) * float(scale)
        pts = tris.reshape(-1, 3)
        uniq, inv = np.unique(pts, axis=0, return_inverse=True)
        faces = inv.reshape(-1, 3).astype(np.int32)
        return MeshData(vertexes=uniq.astype(np.float32), faces=faces)

    raise RuntimeError("Install either 'trimesh' or 'numpy-stl' to load STL files.")


class SerialManager(QtCore.QObject):
    new_attitude = QtCore.pyqtSignal(float, float, float, float)  # (timestamp, pitch, roll, yaw)
    new_setpoints = QtCore.pyqtSignal(float, float)               # (pitch_set, roll_set)
    new_gains = QtCore.pyqtSignal(float, float, float, float, float, float)  # pKp,pKi,pKd,rKp,rKi,rKd
    throttle_status = QtCore.pyqtSignal(bool)  # True=FULL, False=KILL
    status_changed = QtCore.pyqtSignal(str)
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

                    # Throttle status reply via PKT_STICK
                    elif pkt_type == PKT_STICK and len(payload) >= 1:
                        # Accept either [status] or [0x03, status]
                        if len(payload) >= 2 and payload[0] == STICK_QUERY_THROTTLE:
                            status_byte = payload[1]
                        else:
                            status_byte = payload[0]
                        if status_byte in (0x00, 0x01):
                            self.throttle_status.emit(bool(status_byte))

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

    def send_save_to_eeprom(self):
        self.send_packet(PKT_SAVE_EEPROM, b'')

    def send_restore_defaults(self):
        self.send_packet(PKT_RESET_DEFAULTS, b'')

    def send_stick_mode(self, enable: bool):
        self.send_packet(PKT_STICK, bytes([0x01 if enable else 0x00]))

    def send_throttle_status_request(self):
        # Ask FCC for current throttle state
        self.send_packet(PKT_STICK, bytes([STICK_QUERY_THROTTLE]))


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
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.plot.getViewBox().setMouseEnabled(x=True, y=False)
        pg.setConfigOptions(antialias=True, useOpenGL=True)

        self.window_sec = float(WINDOW_SEC)
        self.auto_scroll = False
        self.plot.setXRange(0, self.window_sec, padding=0)

        self.times = deque(maxlen=MAX_LEN)
        self.pitches = deque(maxlen=MAX_LEN)
        self.rolls = deque(maxlen=MAX_LEN)

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
                t_end = t_arr[-1]
                t_start = max(0.0, t_end - self.window_sec)
                mask = (t_arr >= t_start) & (t_arr <= t_end)
                self.plot.setXRange(t_start, t_end, padding=0)
            else:
                mask = t_arr <= self.window_sec

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


# ---------- Dedicated thread to drive 3D at steady FPS ----------
class AttitudeThread(QtCore.QThread):
    update_rpy = QtCore.pyqtSignal(float, float, float)  # pitch, roll, yaw

    def __init__(self, fps=60, alpha=0.25, parent=None):
        super().__init__(parent)
        self.fps = int(max(1, fps))
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._p_est = None
        self._r_est = None
        self._y_est = None

    def push(self, pitch, roll, yaw):
        with self._lock:
            self._latest = (float(pitch), float(roll), float(yaw))

    def run(self):
        interval_ms = int(1000 / self.fps)
        while self._running:
            sample = None
            with self._lock:
                if self._latest is not None:
                    sample = self._latest
            if sample is not None:
                p, r, y = sample
                if self._p_est is None:
                    self._p_est, self._r_est, self._y_est = p, r, y
                else:
                    a = self.alpha
                    self._p_est = a * p + (1 - a) * self._p_est
                    self._r_est = a * r + (1 - a) * self._r_est
                    self._y_est = a * y + (1 - a) * self._y_est
                self.update_rpy.emit(self._p_est, self._r_est, self._y_est)
            self.msleep(interval_ms)

    def stop(self):
        self._running = False


def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = np.pi
    else:
        axis = np.cross(a, b)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = np.arccos(c)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


class Rocket3DWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        TARGET_SIZE = 2.5
        self.SCALE = 1.0
        self._model_scale = 1.0
        self._parts = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_GL:
            lbl = QtWidgets.QLabel("OpenGL (pyqtgraph.opengl) not available")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.setFixedSize(260, 260)
            layout.addWidget(lbl)
            self.view = None
            return

        self.view = GLViewWidget()
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.view.setMinimumSize(260, 260)
        self.setFixedSize(260, 260)

        self.view.setBackgroundColor((255, 255, 255))
        layout.addWidget(self.view)

        self.view.setCameraPosition(distance=4.5, elevation=16, azimuth=40)

        self.axis = GLAxisItem(size=QtGui.QVector3D(2.5, 2.5, 2.5))
        self.view.addItem(self.axis)

        grid = GLGridItem()
        grid.scale(0.6, 0.6, 0.6)
        grid.translate(0, 0, -2.0)
        try:
            grid.setColor((0, 0, 0, 0.18))
        except Exception:
            pass
        self.view.addItem(grid)

        md = None
        try:
            stl_path = os.path.join(os.path.dirname(__file__), STL_FILENAME)
        except Exception:
            stl_path = STL_FILENAME

        if os.path.exists(stl_path):
            try:
                md = meshdata_from_stl(stl_path, scale=1.0)
            except Exception as e:
                print(f"Failed to load STL '{stl_path}': {e}")

        if md is not None:
            V = md.vertexes().astype(np.float32)
            F = md.faces().astype(np.int32)
            center = V.mean(axis=0)
            Vc = V - center

            cov = np.cov(Vc.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal_dir = eigvecs[:, -1]
            R = _rotation_matrix_from_a_to_b(principal_dir, np.array([0.0, 0.0, -1.0], dtype=np.float32))
            Vc = (Vc @ R.T).astype(np.float32)

            z = Vc[:, 2]
            zmin, zmax = float(z.min()), float(z.max())
            span = max(1e-6, (zmax - zmin))
            top_mask = z >= (zmin + 0.85 * span)
            bot_mask = z <= (zmin + 0.15 * span)

            def median_radius(mask):
                pts = Vc[mask]
                if len(pts) == 0:
                    return 0.0
                r = np.linalg.norm(pts[:, :2], axis=1)
                return float(np.median(r)) if len(r) else 0.0

            r_top = median_radius(top_mask)
            r_bot = median_radius(bot_mask)

            if r_bot > r_top:
                Rx_pi = np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]], dtype=np.float32)
                Vc = (Vc @ Rx_pi.T).astype(np.float32)

            z = Vc[:, 2]
            zmin, zmax = float(z.min()), float(z.max())
            height = max(1e-6, (zmax - zmin))
            Vc[:, 2] -= float(PIVOT_RAISE_FRAC) * height

            extents = Vc.max(axis=0) - Vc.min(axis=0)
            max_extent = float(max(1e-9, extents.max()))
            fit_scale = TARGET_SIZE / max_extent
            self._model_scale = fit_scale * self.SCALE

            md_centered = MeshData(vertexes=Vc, faces=F)
            body = GLMeshItem(
                meshdata=md_centered, smooth=True, shader='shaded',
                color=MODEL_OFF_WHITE,
                drawEdges=True, edgeColor=MODEL_EDGE_COLOR
            )
            self.view.addItem(body)
            self._parts.append(body)

        else:
            verts = np.array([
                [ 0.0,   0.0,   1.5],
                [-0.12, -0.12, -1.5],
                [ 0.12, -0.12, -1.5],
                [ 0.12,  0.12, -1.5],
                [-0.12,  0.12, -1.5],
            ], dtype=np.float32)

            zmin, zmax = float(verts[:,2].min()), float(verts[:,2].max())
            height = max(1e-6, (zmax - zmin))
            verts[:, 2] -= float(PIVOT_RAISE_FRAC) * height

            faces = np.array([
                [0,1,2], [0,2,3], [0,3,4], [0,4,1],
                [1,2,3], [1,3,4],
            ], dtype=np.int32)

            mesh = MeshData(vertexes=verts, faces=faces)
            body = GLMeshItem(
                meshdata=mesh, smooth=False, shader='shaded',
                color=MODEL_OFF_WHITE, drawEdges=True, edgeColor=MODEL_EDGE_COLOR
            )
            self.view.addItem(body)
            self._parts.append(body)

            fin_verts = np.array([
                [ 0.0,  0.16, -1.2],
                [ 0.0,  0.30, -1.6],
                [ 0.0,  0.05, -1.6],
            ], dtype=np.float32)
            fin_faces = np.array([[0,1,2]], dtype=np.int32)
            fin_mesh = MeshData(vertexes=fin_verts, faces=fin_faces)
            fin1 = GLMeshItem(meshdata=fin_mesh, smooth=False, shader='shaded',
                              color=FIN_OFF_WHITE)
            self.view.addItem(fin1)
            self._parts.append(fin1)

            fin_verts2 = fin_verts.copy(); fin_verts2[:,1] *= -1
            fin_mesh2 = MeshData(vertexes=fin_verts2, faces=fin_faces)
            fin2 = GLMeshItem(meshdata=fin_mesh2, smooth=False, shader='shaded',
                              color=FIN_OFF_WHITE)
            self.view.addItem(fin2)
            self._parts.append(fin2)

            self._model_scale = 2.2

        self._apply_transform([])

    def _apply_transform(self, ops):
        for item in self._parts:
            item.resetTransform()
            s = float(self._model_scale)
            item.scale(s, s, s)
            for op in ops:
                op(item)

    def set_attitude(self, pitch_deg: float, roll_deg: float):
        if not self._parts:
            return
        pitch = float(pitch_deg)
        roll  = float(roll_deg)
        self._apply_transform([
            lambda item: item.rotate(pitch, 0, 1, 0),
            lambda item: item.rotate(roll,  1, 0, 0),
        ])

    def set_attitude_rpy(self, roll_deg: float, pitch_deg: float, yaw_deg: float):
        if not self._parts:
            return
        self._apply_transform([
            lambda item: item.rotate(roll_deg,  1, 0, 0),
            lambda item: item.rotate(pitch_deg, 0, 1, 0),
            lambda item: item.rotate(yaw_deg,   0, 0, 1),
        ])


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
        self.throttle_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold;")
        self.throttle_state = False
        self.awaiting_throttle_status = False

        # Stick Mode button (PID assist)
        self.stick_btn = QtWidgets.QPushButton("Stick: Disabled")
        self.stick_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stick_btn.setEnabled(False)
        self.stick_enabled = False

        # Throttle status indicator chip
        self.throttle_state_lbl = QtWidgets.QLabel("Throttle: —")
        self.throttle_state_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.throttle_state_lbl.setMinimumWidth(110)
        self._set_throttle_indicator("Throttle: —", "#CCCCCC")  # gray

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(com_label)
        control_layout.addWidget(self.com_combo)
        control_layout.addWidget(baud_label)
        control_layout.addWidget(self.baud_combo)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.throttle_btn)
        control_layout.addWidget(self.stick_btn)
        control_layout.addWidget(self.throttle_state_lbl)
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

        # EEPROM buttons
        eeprom_btn_row = QtWidgets.QHBoxLayout()
        self.btn_save_eeprom = QtWidgets.QPushButton("Save to EEPROM")
        self.btn_restore_defaults = QtWidgets.QPushButton("Restore Defaults")
        eeprom_btn_row.addWidget(self.btn_save_eeprom)
        eeprom_btn_row.addWidget(self.btn_restore_defaults)
        eeprom_btn_row.addStretch()

        # X-axis controls
        xaxis_row = QtWidgets.QHBoxLayout()
        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll X")
        self.chk_autoscroll.setChecked(False)
        self.chk_autoscroll.setEnabled(False)
        self.win_sec_spin = QtWidgets.QDoubleSpinBox()
        self.win_sec_spin.setLocale(latin_locale)
        self.win_sec_spin.setRange(0.5, 120.0)
        self.win_sec_spin.setDecimals(1)
        self.win_sec_spin.setSingleStep(0.5)
        self.win_sec_spin.setValue(float(WINDOW_SEC))
        self.win_sec_spin.setSuffix(" s")
        self.win_sec_spin.setEnabled(False)
        xaxis_row.addWidget(self.chk_autoscroll)
        xaxis_row.addWidget(QtWidgets.QLabel("Window:"))
        xaxis_row.addWidget(self.win_sec_spin)
        xaxis_row.addStretch()

        # ----------- 3D Rocket (top-right) -----------
        self.rocket_widget = Rocket3DWidget()

        # ----------- Plot -----------
        self.plot_widget = PIDPlotWidget()

        # ----------- Status Bar -----------
        self.status = QtWidgets.QLabel("Disconnected")

        # ----------- Compose Layouts -----------
        left_controls = QtWidgets.QVBoxLayout()
        left_controls.addLayout(control_layout)
        left_controls.addLayout(sp_layout)
        left_controls.addWidget(gp_pitch)
        left_controls.addWidget(gp_roll)
        left_controls.addLayout(gains_btn_row)
        left_controls.addLayout(eeprom_btn_row)
        left_controls.addLayout(xaxis_row)
        left_controls.addStretch(1)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addLayout(left_controls, 1)
        top_row.addWidget(self.rocket_widget, 0, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(top_row)
        main_layout.addWidget(self.plot_widget, 1)
        main_layout.addWidget(self.status)

        # ----------- Serial Manager ---------
        self.serial_manager = SerialManager()
        self.serial_manager.new_attitude.connect(self.handle_attitude)
        self.serial_manager.new_setpoints.connect(self.handle_new_setpoints)
        self.serial_manager.new_gains.connect(self.handle_new_gains)
        self.serial_manager.throttle_status.connect(self.on_throttle_status_from_fcc)
        self.serial_manager.status_changed.connect(self.on_status_changed)
        self.serial_manager.ack_saved.connect(self.on_ack_saved)
        self.serial_manager.ack_restored.connect(self.on_ack_restored)

        # ----------- 3D Attitude Thread -----------
        self.att_thread = AttitudeThread(fps=60, alpha=0.25)
        self.att_thread.update_rpy.connect(self.on_3d_update)
        self.att_thread.start()

        # ----------- Timer for plot -----------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.plot_widget.update_plot)
        self.timer.start(TIMER_MS)

        # ----------- Buttons -----------
        self.connect_btn.clicked.connect(self.on_connect_toggle)
        self.throttle_btn.clicked.connect(self.on_throttle_clicked)
        self.stick_btn.clicked.connect(self.on_stick_clicked)
        self.btn_request_setpoints.clicked.connect(self.on_request_setpoints_clicked)
        self.btn_set_setpoints.clicked.connect(self.on_set_setpoints_clicked)
        self.btn_request_gains.clicked.connect(self.on_request_gains_clicked)
        self.btn_set_gains.clicked.connect(self.on_set_gains_clicked)
        self.btn_save_eeprom.clicked.connect(self.on_save_eeprom_clicked)
        self.btn_restore_defaults.clicked.connect(self.on_restore_defaults_clicked)
        self.chk_autoscroll.toggled.connect(self.plot_widget.set_auto_scroll)
        self.win_sec_spin.valueChanged.connect(self.plot_widget.set_window_sec)

        # ----------- Load settings ----------
        self.load_settings()

    # ---------- small helpers ----------
    def _set_throttle_indicator(self, text: str, bg: str):
        self.throttle_state_lbl.setText(text)
        self.throttle_state_lbl.setStyleSheet(
            f"QLabel {{ background-color: {bg}; color: #111; padding: 2px 6px; "
            f"border-radius: 6px; font-weight: bold; }}"
        )

    def set_status(self, msg: str):
        self.status.setText(msg)

    # -------------- Helpers for Throttle & Stick UI ----------------
    def set_throttle_ui(self, full: bool):
        """Apply throttle state to UI and related controls (no command send)."""
        self.throttle_state = bool(full)
        if self.throttle_state:
            self.throttle_btn.setText("Kill Throttle")
            self.throttle_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.stick_btn.setEnabled(True)
            self._set_throttle_indicator("Throttle: FULL", "#93C47D")  # green
        else:
            self.throttle_btn.setText("Full Throttle")
            self.throttle_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
            # If stick mode is on, force it off
            if self.stick_enabled:
                try:
                    self.serial_manager.send_stick_mode(False)
                except Exception:
                    pass
                self.stick_enabled = False
                self.stick_btn.setText("Stick: Disabled")
                self.stick_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.stick_btn.setEnabled(False)
            self._set_throttle_indicator("Throttle: KILL", "#EA9999")  # red

    def query_throttle_status(self):
        """Request throttle status from FCC and lock controls until reply arrives."""
        self.awaiting_throttle_status = True
        self.throttle_btn.setEnabled(False)
        self.stick_btn.setEnabled(False)
        self.set_status("Querying throttle status...")
        self._set_throttle_indicator("Throttle: …", "#FFD966")  # yellow
        try:
            self.serial_manager.send_throttle_status_request()
        except Exception:
            pass

    # ---------------- Qt Events & Slots ----------------
    def closeEvent(self, event):
        try:
            self.serial_manager.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'att_thread') and self.att_thread.isRunning():
                self.att_thread.stop()
                self.att_thread.wait(1000)
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

    # Serial status changes (connect/disconnect/errors)
    def on_status_changed(self, msg: str):
        self.set_status(msg)
        if msg == "Connected":
            # On connect, ask FCC for throttle state and wait
            self.query_throttle_status()
        elif msg == "Disconnected" or msg.startswith("Error:"):
            # When disconnected or error, lock controls & show unknown state
            self.awaiting_throttle_status = False
            self.throttle_btn.setEnabled(False)
            self.stick_btn.setEnabled(False)
            self.stick_enabled = False
            self.stick_btn.setText("Stick: Disabled")
            self.stick_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.throttle_btn.setText("Full Throttle")
            self.throttle_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold;")
            self._set_throttle_indicator("Throttle: —", "#CCCCCC")  # gray

    def on_connect_toggle(self):
        if self.serial_manager.ser and self.serial_manager.ser.is_open:
            # Disconnect
            self.serial_manager.close()
            self.connect_btn.setText("Connect")
            self.plot_widget.reset_data()
            self.chk_autoscroll.setEnabled(False)
            self.win_sec_spin.setEnabled(False)
        else:
            # Connect
            self.plot_widget.reset_data()
            port = self.com_combo.currentText()
            baud = int(self.baud_combo.currentText())
            if port and self.serial_manager.open(port, baud):
                self.connect_btn.setText("Disconnect")
                self.chk_autoscroll.setEnabled(True)
                self.win_sec_spin.setEnabled(True)
                # Buttons remain disabled until throttle status reply arrives
                self.throttle_btn.setEnabled(False)
                self.stick_btn.setEnabled(False)
                # Indicator will be set to "…" by query_throttle_status() once "Connected" arrives

    def on_throttle_status_from_fcc(self, is_full: bool):
        """FCC replied with throttle state -> update UI and unlock controls."""
        self.set_status(f"Throttle status from FCC: {'FULL' if is_full else 'KILL'}")
        self.set_throttle_ui(is_full)
        self.awaiting_throttle_status = False
        self.throttle_btn.setEnabled(True)

    def handle_attitude(self, now, pitch, roll, yaw):
        self.plot_widget.add_data(now, pitch, roll, yaw)
        if hasattr(self, 'att_thread'):
            self.att_thread.push(pitch, roll, yaw)

    @QtCore.pyqtSlot(float, float, float)
    def on_3d_update(self, pitch, roll, yaw):
        try:
            self.rocket_widget.set_attitude(pitch, -roll)
        except Exception:
            pass

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
        if self.awaiting_throttle_status:
            QtWidgets.QMessageBox.information(self, "Please wait", "Waiting for throttle status from FCC...")
            return

        if not self.throttle_state:  # Full Throttle
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm", "Send FULL THROTTLE command?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm == QtWidgets.QMessageBox.Yes:
                self.serial_manager.send_packet(PKT_THROTTLE, bytes([0x01]))
                # Do not assume—ask FCC for authoritative state
                self.query_throttle_status()
        else:  # Kill Throttle
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm", "Send KILL THROTTLE command?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm == QtWidgets.QMessageBox.Yes:
                self.serial_manager.send_packet(PKT_THROTTLE, bytes([0x00]))
                # Do not assume—ask FCC for authoritative state
                self.query_throttle_status()

    def on_stick_clicked(self):
        if not self.serial_manager.ser or not self.serial_manager.ser.is_open:
            QtWidgets.QMessageBox.warning(self, "Warning", "Serial not connected!")
            return
        if self.awaiting_throttle_status:
            QtWidgets.QMessageBox.information(self, "Please wait", "Waiting for throttle status from FCC...")
            return
        if not self.throttle_state:
            QtWidgets.QMessageBox.warning(self, "Warning", "Stick Mode requires FULL THROTTLE.")
            return

        if not self.stick_enabled:
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm",
                "Enable Stick Mode (PID Assist)?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm == QtWidgets.QMessageBox.Yes:
                self.serial_manager.send_stick_mode(True)
                self.stick_enabled = True
                self.stick_btn.setText("Stick: Enabled")
                self.stick_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
                self.set_status("Stick mode ENABLE command sent.")
        else:
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm",
                "Disable Stick Mode?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm == QtWidgets.QMessageBox.Yes:
                self.serial_manager.send_stick_mode(False)
                self.stick_enabled = False
                self.stick_btn.setText("Stick: Disabled")
                self.stick_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                self.set_status("Stick mode DISABLE command sent.")

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
