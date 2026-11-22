"""Load a saved physical loop and its measured residual-stream handoff."""
from .atlas_stream_handoff import install_program
from .modeling_sensenova_loop import SenseNovaLoopModel
from .modeling_sensenova_loop import SenseNovaLoopForConditionalGeneration

def _install(model, config):
    text, = (m for m in model.modules() if hasattr(m, '_loop_plan') and hasattr(m, 'layers'))
    install_program(text, config.atlas_loop_handoff)

class AtlasSenseNovaLoopModel(SenseNovaLoopModel):
    def __init__(self, config):
        super().__init__(config)
        _install(self, config)

class AtlasSenseNovaLoopForConditionalGeneration(SenseNovaLoopForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        _install(self, config)
