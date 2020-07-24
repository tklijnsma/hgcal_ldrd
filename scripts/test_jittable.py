import sys, os.path as osp, os
import torch

# Set the environment variable that is used as a guard for importing the jittable model
os.environ['HGCAL_JITTABLE'] = 'True'

THIS_DIR = osp.abspath(osp.dirname(__file__))
sys.path.append(osp.join(THIS_DIR, '../src'))

import edgenetscript
script = edgenetscript.TrainingScript(debug=False)

script.prevent_reprocessing = True
script.model_name = 'EdgeNetWithCategoriesJittable'
script.load_checkpoint = '../saved_weights/models/model_checkpoint_EdgeNetWithCategories_264403_5b5c05404f_csharma.best.pth.tar'

trainer = script.get_trainer()
print(trainer.model)

model = torch.jit.script(trainer.model)

