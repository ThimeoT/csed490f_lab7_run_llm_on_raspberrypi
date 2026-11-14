# Lab 7 - Run LLM on Raspberry Pi
CSED490F - Deep Learning Implementations  

**Requirements**  
+ fastapi>=0.115
+ uvicorn[standard]>=0.30
+ ollama>=0.4
+ requests>=2.31

**Lauch on Raspberry Pi**  
```bash
ollama list                 # verifies that llama3.2 is here
./run.sh                    # or : CHAT_MODEL=llama3.2:1b ./run.sh
hostname -I                 # to get the IP of Raspberry Pi
```

**Launch from another device**
```bash
curl http://<pi-ip>:8000/health
curl -N -X POST http://<pi-ip>:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"why is the sky blue?"}'
curl -X POST http://<pi-ip>:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"hi","stream":false}'
python client/chat_cli.py --url http://<pi-ip>:8000
python client/benchmark.py --url http://<pi-ip>:8000 -n 10 -c 1
python client/benchmark.py --url http://<pi-ip>:8000 -n 10 -c 3   # load test
```

In a web browser `http://<pi-ip>:8000/` opens the chat and `http://<pi-ip>:8000/docs` the Swagger doc 