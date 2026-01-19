# Script untuk menjalankan dashboard
import subprocess
import sys

def main():
    """Menjalankan dashboard Streamlit"""
    try:
        print("🚀 Memulai Dashboard Monitoring Titik Panas...")
        print("📊 Dashboard akan terbuka di browser pada http://localhost:8501")
        print("⏹️  Tekan Ctrl+C untuk menghentikan dashboard")
        
        # Menjalankan streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard_titik_panas.py",
            # "--server.port=8578",
            # "--server.address=localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n✅ Dashboard dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error menjalankan dashboard: {e}")

if __name__ == "__main__":
    main()
