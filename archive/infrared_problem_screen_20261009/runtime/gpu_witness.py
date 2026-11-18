import json, torch, numpy as np, cv2, scipy
from PIL import Image
torch.manual_seed(46)
torch.cuda.manual_seed_all(46)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
a=torch.randn(8,8,device="cuda")
b=a@a.T
assert torch.isfinite(b).all()
result={"sentinel":"WITNESS_OK","seed":46,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"cv2":cv2.__version__,"device":torch.cuda.get_device_name(),"shape":list(b.shape),"sum":b.sum().item()}
print(json.dumps(result))
