import re
from os.path import abspath, dirname

operator_mapping = {
    '>': '>',
    '>=': '>=',
    '<': '<',
    '<=': '<=',
    '==': '==',
    '<>': '!='
}

def parse_constraints(line):
    match = re.search(r'\{(.+?)\}', line)
    if match:
        inner_content = match.group(1)
        constraints = inner_content.split('&&')
        result_list = []
        for constraint in constraints:
            constraint = constraint.strip()
            for op in ['<>','<=','>=','==','<','>']:
                if op in constraint:
                    left, right = constraint.split(op, 1)
                    A1 = left.split('.')[-1].strip()  
                    B = operator_mapping[op]          
                    result_list.append([1, A1, B, 2, A1])
                    break
        return result_list
    return []

def txt2cons(path=dirname(dirname(abspath(__file__)))+'/data/CONS.txt',if_print=False):
    with open(path, 'r', encoding='utf-8') as file:
        id,constraint=1,{}
        for line in file:
            results = parse_constraints(line)
            constraint[f'DC{id}']=results
            id+=1
    constraint={k:constraint[k] for k in constraint.keys() if len(constraint[k])>0}
    if if_print:
        for p in constraint.values():
            print(p)
    return constraint




