import ast
import json
from pathlib import Path
import pytest
from cnn_lab.model import Config,MAX_EPOCHS,validate
from cnn_lab.config_export import config_json

@pytest.mark.parametrize('epochs,accepted',[(200,True),(201,False),(0,False)])
def test_epoch_boundaries(epochs,accepted):
    config=Config(epochs=epochs)
    if accepted:validate(config)
    else:
        with pytest.raises(ValueError,match='1–200 épocas'):validate(config)


def test_two_hundred_json_roundtrip_defaults_and_ui_limits():
    config=Config(epochs=200)
    assert MAX_EPOCHS==200 and Config.from_dict(json.loads(config_json(config))).epochs==200
    assert Config().epochs==8 and Config().patience==3
    tree=ast.parse((Path(__file__).resolve().parents[1]/'app.py').read_text())
    inputs={call.args[0].value:call for call in ast.walk(tree) if isinstance(call,ast.Call) and call.args and isinstance(call.args[0],ast.Constant) and call.args[0].value in ('Máximo de épocas','Épocas por fold','Paciencia · épocas')}
    for label in ('Máximo de épocas','Épocas por fold'):assert isinstance(inputs[label].args[2],ast.Name) and inputs[label].args[2].id=='MAX_EPOCHS'
    assert inputs['Paciencia · épocas'].args[2].value==30
