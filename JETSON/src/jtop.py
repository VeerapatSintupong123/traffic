from jtop import jtop
import threading
import time

class JTopMonitor:
    def __init__(self, interval=0.5):
        self.jetson = None
        self.stats_history = []
        self.monitoring = False
        self._thread = None
        self.interval = interval

        try:
            self.jetson = jtop()
        except ImportError:
            print("jtop library not found. System resource monitoring will be disabled.")
        except Exception as e:
            print(f"Could not initialize jtop: {e}")
    
    def _collect_stats(self, jetson):
        """Callback to collect stats from jtop"""
        if self.monitoring:
            stats = jetson.stats
            log_entry = {
                'time': str(stats.get('time')),
                'gpu': stats.get('GPU'),
                'ram': stats.get('RAM'),
                'swap': stats.get('SWAP'),
                'iram': stats.get('IRAM'),
                'CPU1': stats.get('CPU1'),
                'CPU2': stats.get('CPU2'),
                'CPU3': stats.get('CPU3'),
                'CPU4': stats.get('CPU4'),
                'Temp_AO': stats.get('Temp AO'),
                'Temp_CPU': stats.get('Temp CPU'),
                'Temp_GPU': stats.get('Temp GPU'),
                'Temp_PLL': stats.get('Temp PLL'),
                'Temp_thermal': stats.get('Temp thermal'),
                'PW_CPU': stats.get('Power POM_5V_CPU'),
                'PW_GPU': stats.get('Power POM_5V_GPU'),
                'PW_total': stats.get('Power TOT'),
            }
            self.stats_history.append(log_entry)

    def start(self):
        """Start monitoring in a separate thread"""
        if not self.jetson:
            return
        
        self.monitoring = True
        self.stats_history = []
        
        def monitor_loop():
            with self.jetson:
                self.jetson.attach(self._collect_stats)
                while self.monitoring:
                    time.sleep(self.interval)
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_stats(self):
        """Get collected statistics"""
        return self.stats_history.copy()

    def clear_stats(self):
        """Clear collected statistics"""
        self.stats_history = []