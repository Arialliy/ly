from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess

out = Path(__file__).resolve().parent
root = out.parents[1]
path = "/p/experiments/" + out.name
docker_image = "sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92"
command = ["docker","run","--rm","--network","none","--cpus","4","--memory","6g",
           "--user",f"{os.getuid()}:{os.getgid()}","-e","PYTHONDONTWRITEBYTECODE=1",
           "-e","PYTHONPATH=/p/environment/site:"+path+"/runtime",
           "-v",str(root)+":/p:ro","-v",str(out)+":"+path+":rw",
           "--entrypoint","/opt/conda/bin/python",docker_image,path+"/count_control.py"]
receipt = {"start":datetime.now(timezone.utc).isoformat(),"command":command}
(out/"launch.json").write_text(json.dumps(receipt,indent=2)+"\n")
with (out/"run.log").open("w") as stream:
    job = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
receipt.update(end=datetime.now(timezone.utc).isoformat(),exit_code=job.returncode)
(out/"launch.json").write_text(json.dumps(receipt,indent=2)+"\n")
raise SystemExit(job.returncode)
