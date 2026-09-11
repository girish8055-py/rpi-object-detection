import logging
import threading
import time

logger = logging.getLogger("GPS-Telemetry")

# Safely import pymavlink
HAS_MAVLINK = False
try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    logger.warning("pymavlink package missing. Install: pip install pymavlink")


class TelemetryManager:
    """Manages background Flight Controller (FC) and GPS telemetry connections (MAVLink / Serial NMEA / UDP).
    
    Automatically listens for MAVLink heartbeats on USB (/dev/ttyACM0, /dev/ttyUSB0) or GPIO UART ports.
    If no FC/GPS is connected, latitude and longitude default to None (null in JSON).
    Reconnection and heartbeat listening happen continuously in the background.
    """

    def __init__(self, connection_str=None, baudrate=None):
        self.connection_str = connection_str
        self.user_baudrate = baudrate
        
        self.latitude = None
        self.longitude = None
        self.altitude_m = None
        self.gps_fix_type = 0
        self.connected = False
        
        self._running = True
        self._thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self._thread.start()

    def _telemetry_worker(self):
        """Background worker searching continuously for FC heartbeat signals."""
        while self._running:
            if self.connection_str:
                ports_to_try = [self.connection_str]
            else:
                ports_to_try = [
                    "/dev/ttyACM0",       # USB CDC Flight Controller (Pixhawk/ArduPilot/Cube)
                    "/dev/ttyUSB0",       # USB FTDI adapter / external GPS
                    "/dev/ttyAMA0",       # RPi GPIO UART
                    "udp:127.0.0.1:14550" # Local MAVProxy / telemetry mirror
                ]

            bauds_to_try = [self.user_baudrate] if self.user_baudrate else [57600, 115200, 921600, 115200]
            connected_device = None

            if HAS_MAVLINK:
                for port in ports_to_try:
                    if connected_device:
                        break
                    
                    # UDP doesn't use baud rate
                    baud_list = [57600] if port.startswith("udp:") else bauds_to_try
                    
                    for baud in baud_list:
                        try:
                            logger.debug(f"Listening for MAVLink heartbeat on '{port}' (baud={baud})...")
                            mav = mavutil.mavlink_connection(port, baud=baud, timeout=0.8)
                            
                            # Listen for FC heartbeat
                            msg = mav.wait_heartbeat(timeout=1.2)
                            if msg:
                                logger.info(f"Connected! Received MAVLink heartbeat from Flight Controller on '{port}' (baud={baud}) [SysID={mav.target_system}]")
                                connected_device = mav
                                self.connected = True
                                break
                        except Exception:
                            continue

            if connected_device:
                self._read_mavlink_stream(connected_device)
            else:
                self.latitude = None
                self.longitude = None
                self.altitude_m = None
                self.connected = False
                time.sleep(2.5)  # Pause before next scan iteration

    def _read_mavlink_stream(self, mav):
        """Streams GPS messages from FC while connection remains active."""
        try:
            # Request GPS position data streams (5 Hz rate)
            mav.mav.request_data_stream_send(
                mav.target_system,
                mav.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                5, 1
            )

            while self._running:
                msg = mav.recv_match(type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], blocking=True, timeout=2.0)
                if not msg:
                    logger.warning("No telemetry heartbeat/data received for 2 seconds. Re-scanning for FC connection...")
                    break
                
                msg_type = msg.get_type()
                if msg_type == 'GLOBAL_POSITION_INT':
                    self.latitude = round(msg.lat / 1e7, 7)
                    self.longitude = round(msg.lon / 1e7, 7)
                    self.altitude_m = round(msg.relative_alt / 1000.0, 2)
                elif msg_type == 'GPS_RAW_INT':
                    if msg.fix_type >= 2:  # 2D or 3D fix
                        self.latitude = round(msg.lat / 1e7, 7)
                        self.longitude = round(msg.lon / 1e7, 7)
                        self.altitude_m = round(msg.alt / 1000.0, 2)
                        self.gps_fix_type = msg.fix_type
        except Exception as e:
            logger.error(f"Flight Controller stream error: {e}")
        finally:
            self.connected = False
            self.latitude = None
            self.longitude = None
            self.altitude_m = None
            try:
                mav.close()
            except Exception:
                pass

    def get_telemetry(self):
        """Returns current GPS telemetry dict."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "fc_connected": self.connected
        }

    def stop(self):
        self._running = False
