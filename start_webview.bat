if not exist "AuidoReactiveLedStrip\Application\bin\Debug\net8.0" (
    mkdir "AuidoReactiveLedStrip\Application\bin\Debug\net8.0"
)

copy /y "audio_parameter_web_ui.py" "AuidoReactiveLedStrip\Application\bin\Debug\net8.0\"
cd "AuidoReactiveLedStrip\Application\bin\Debug\net8.0\"
streamlit run audio_parameter_web_ui.py --server.address 0.0.0.0 --server.port 8501