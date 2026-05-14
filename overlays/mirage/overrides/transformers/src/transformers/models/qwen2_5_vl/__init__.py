# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


import os

if TYPE_CHECKING:
    from .configuration_qwen2_5_vl import *
    if os.environ.get("USE_MIRAGE_NEW") == "1":
        from .modeling_qwen2_5_vl_new import *
    else:
        from .modeling_qwen2_5_vl import *
    from .processing_qwen2_5_vl import *
else:
    import sys

    _file = globals()["__file__"]
    import_structure = define_import_structure(_file)
    
    if os.environ.get("USE_MIRAGE_NEW") == "1":
        # Swap modeling_qwen2_5_vl with modeling_qwen2_5_vl_new
        if "modeling_qwen2_5_vl" in import_structure:
            import_structure["modeling_qwen2_5_vl_new"] = import_structure.pop("modeling_qwen2_5_vl")
            
    sys.modules[__name__] = _LazyModule(__name__, _file, import_structure, module_spec=__spec__)
