from dataclasses import dataclass
from typing import List, Dict

def replace_speical_span(text, contents: Dict[str, str]):
    # 不用format，而是手动替换。因为format用到的大括号真的太常见了。
    replaced_text = text
    for k, v in contents.items():
        replaced_text = replaced_text.replace(f"{{{k}}}", v)
    return replaced_text

# 定义各级的输入格式
@dataclass
class Segment:
    text: str
    def fill_template(self, contents: Dict[str, str]):
        return Segment(replace_speical_span(self.text, contents))

@dataclass
class Block:
    segments: List[Segment]
    def fill_template(self, contents: Dict[str, str]):
        return Block([s.fill_template(contents) for s in self.segments])

# 黑盒模型类：支持输入contents（如果是str类型就直接认为是question），输出answer
class ModelBox:
    def __init__(self, infer_template: Block):
        self.infer_template = infer_template
    def _predict(self, input: Block):
        '''
        需要实现：将input转换为模型输入，生成答案（涉及到解码策略），并从生成的文本中提取答案（涉及到答案的格式）。
        让子类实现。
        '''
        raise NotImplementedError
    def predict(self, contents: str | Dict[str, str]) -> str:
        if isinstance(contents, str):
            contents = {"question": contents}
        input = self.infer_template.fill_template(contents)
        return self._predict(input)
    def _loss_evaluate(self, input: Block) -> float:
        raise NotImplementedError
    def loss_evaluate(self, contents: Dict[str, str]) -> float:
        '''
        contents中要包含question和answer
        '''
        input = self.infer_template.fill_template(contents)
        return self._loss_evaluate(input, contents["answer"])
    
# 模型创建者：包含基座模型，能根据context生成ModelBox
class ModelCreator:
    def __init__(self):
        pass
    def _build(self, contents: Dict[str, str]):
        '''
        需要实现：根据contents生成ModelBox
        因为修改通常会直接加在基座模型上，所以注意要备份基座模型，后面可能要通过self.recover()恢复。
        让子类实现。
        '''
        raise NotImplementedError
    def recover(self):
        '''
        恢复基座模型
        '''
        raise NotImplementedError
    def build(self, contents: str | Dict[str, str]):
        if isinstance(contents, str):
            contents = {"contents": {"context": contents}}
        return self._build(contents)