"""Measure heat-up and cool-down time constants under sustained LLM inference."""
import pynvml, time, json, threading, urllib.request, csv, sys, math
pynvml.nvmlInit(); H=pynvml.nvmlDeviceGetHandleByIndex(0)
MODEL=sys.argv[1]; LOAD=float(sys.argv[2]); COOL=float(sys.argv[3]); TAG=sys.argv[4]
IDLE_C=float(sys.argv[5]) if len(sys.argv)>5 else 60.0

def T(): return pynvml.nvmlDeviceGetTemperature(H,0)
def CLK(): return pynvml.nvmlDeviceGetClockInfo(H,1)
def PW(): return pynvml.nvmlDeviceGetPowerUsage(H)/1000
def THR(): return pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(H)

tel=[]; phase=["idle"]; stop=False
def logger():
    t0=time.time()
    while not stop:
        try: tel.append((round(time.time()-t0,3),phase[0],T(),PW(),CLK(),THR()))
        except Exception: pass
        time.sleep(0.05)

# --- 1. cool to defined cold start ---
print(f"[{TAG}] waiting for cold start <{IDLE_C}C (now {T()}C)...",flush=True)
w=time.time()
while T()>IDLE_C and time.time()-w<900: time.sleep(5)
print(f"[{TAG}] cold start reached: {T()}C clk={CLK()}MHz after {time.time()-w:.0f}s",flush=True)

th=threading.Thread(target=logger,daemon=True); th.start()
time.sleep(10)  # baseline idle

# --- 2. sustained load (heat-up) ---
phase[0]="load"
P="Describe the architecture of a distributed cloud system and its trade-offs in detail."
reqs=[]; t0=time.time(); i=0
while time.time()-t0 < LOAD:
    body=json.dumps({"model":MODEL,"prompt":P,"stream":False,
        "options":{"num_predict":128,"temperature":0.0}}).encode()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(
          "http://127.0.0.1:11434/api/generate",body,{"Content-Type":"application/json"}),timeout=300))
    except Exception as e: print("ERR",e,flush=True); break
    el=time.time()-t0; n=r.get("eval_count",0); ed=r.get("eval_duration",1)/1e9
    reqs.append((el,n,ed,n/ed if ed else 0)); i+=1
    if i%10==0: print(f"  [{TAG}] load t={el:5.0f}s {n/ed:5.1f}tok/s {T()}C {CLK()}MHz",flush=True)

# --- 3. cool-down (idle) ---
phase[0]="cool"; c0=time.time()
print(f"[{TAG}] LOAD OFF at {T()}C {CLK()}MHz -- cooling {COOL}s",flush=True)
while time.time()-c0 < COOL: time.sleep(1)
stop=True; time.sleep(0.3)

csv.writer(open(f"tc_{TAG}_tel.csv","w",newline="")).writerows(
    [("t","phase","temp","power","sm_mhz","throttle")]+tel)
csv.writer(open(f"tc_{TAG}_reqs.csv","w",newline="")).writerows(
    [("elapsed","tokens","decode_s","tok_s")]+reqs)
print(f"[{TAG}] DONE reqs={len(reqs)} tel={len(tel)}",flush=True)
