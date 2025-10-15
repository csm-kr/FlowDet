
import rootutils
rootutils.setup_root(__file__, indicator='.vscode', pythonpath=True)

from flowdet import FlowDet

def build_model(args):
    model = FlowDet(args=args)
    return model
