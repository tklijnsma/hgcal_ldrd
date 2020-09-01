import sys, os.path as osp, os
import torch

THIS_DIR = osp.abspath(osp.dirname(__file__))
sys.path.append(osp.join(THIS_DIR, '../src'))
from models import DynamicReductionNetworkJittable
nonjit_model = DynamicReductionNetworkJittable(input_dim=5, hidden_dim=64, output_dim=1, k=16)
model = torch.jit.script(nonjit_model)

print(model)

# weights_file = '../saved_weights/models/model_checkpoint_DynamicReductionNetwork_264403_5b5c05404f_csharma.best.pth.tar'
# pretrained_dict = torch.load(weights_file, map_location=torch.device('cpu'))['model']

# model_dict = model.state_dict()

# map_pretrained_to_jitmodel = {
#     'nodenetwork0.nn.0.weight' : 'firstnodenetwork.nn.0.weight',
#     'nodenetwork0.nn.0.bias'   : 'firstnodenetwork.nn.0.bias',
#     'nodenetwork0.nn.2.weight' : 'firstnodenetwork.nn.2.weight',
#     'nodenetwork0.nn.2.bias'   : 'firstnodenetwork.nn.2.bias',
#     'nodenetwork1.nn.0.weight' : 'nodenetwork.0.nn.0.weight',
#     'nodenetwork1.nn.0.bias'   : 'nodenetwork.0.nn.0.bias',
#     'nodenetwork1.nn.2.weight' : 'nodenetwork.0.nn.2.weight',
#     'nodenetwork1.nn.2.bias'   : 'nodenetwork.0.nn.2.bias',
#     'nodenetwork2.nn.0.weight' : 'nodenetwork.1.nn.0.weight',
#     'nodenetwork2.nn.0.bias'   : 'nodenetwork.1.nn.0.bias',
#     'nodenetwork2.nn.2.weight' : 'nodenetwork.1.nn.2.weight',
#     'nodenetwork2.nn.2.bias'   : 'nodenetwork.1.nn.2.bias',
#     'nodenetwork3.nn.0.weight' : 'nodenetwork.2.nn.0.weight',
#     'nodenetwork3.nn.0.bias'   : 'nodenetwork.2.nn.0.bias',
#     'nodenetwork3.nn.2.weight' : 'nodenetwork.2.nn.2.weight',
#     'nodenetwork3.nn.2.bias'   : 'nodenetwork.2.nn.2.bias',
#     'nodenetwork4.nn.0.weight' : 'nodenetwork.3.nn.0.weight',
#     'nodenetwork4.nn.0.bias'   : 'nodenetwork.3.nn.0.bias',
#     'nodenetwork4.nn.2.weight' : 'nodenetwork.3.nn.2.weight',
#     'nodenetwork4.nn.2.bias'   : 'nodenetwork.3.nn.2.bias',
#     'nodenetwork5.nn.0.weight' : 'nodenetwork.4.nn.0.weight',
#     'nodenetwork5.nn.0.bias'   : 'nodenetwork.4.nn.0.bias',
#     'nodenetwork5.nn.2.weight' : 'nodenetwork.4.nn.2.weight',
#     'nodenetwork5.nn.2.bias'   : 'nodenetwork.4.nn.2.bias',
#     }

# for key in pretrained_dict:
#     model_key = map_pretrained_to_jitmodel.get(key, key)

#     if not model_key in model_dict:
#         print('Skipping key {0}, not in model_dict'.format(model_key))
#         continue

#     if key != model_key:
#         print('Mapping {0} --> {1}'.format(key, model_key))
    
#     model_dict[model_key] = pretrained_dict[key]
    
# model.load_state_dict(model_dict)
# torch.jit.save(model, 'edgenetwithcats.pt')
