from init_demo import add, whisper
from init_demo.str_ops.ops import shout

from init_demo import *

ns = {}
exec("from init_demo import *", ns)
star_imported = sorted(k for k in ns.keys() if not k.startswith("__"))
print("`from init_demo import *`实际导入内容: ", star_imported)
