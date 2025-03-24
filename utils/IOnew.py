import numpy as np
from typing import Iterable,Tuple,List,Mapping
import zlib
import os
import re

class Writer:
    def __init__(self,H:int,W:int,filename:str,mode:int='normal',comments:List[str]=[],compress:bool=False):
        self.filename = filename
        self.mode = mode
        self.compress = compress
        self.comments = comments
        self.Xs = []
        self.Ys = []
        self.shape = H,W
    def __enter__(self):
        return self
    def __setitem__(self,coord:Tuple[int,int],value:int):
        if 1 != value:
            return
        x,y = coord
        assert 0 <= x < self.shape[0] and 0 <= y < self.shape[1], f'Index out of range: {coord}'
        self.Xs.append(x)
        self.Ys.append(y)
    def __exit__(self,exc_type,exc_value,traceback):
        buffer = ''
        self.comments.append(f'H:{self.shape[0]},W:{self.shape[1]}')
        for comment in self.comments:
            for line in comment.split('\n'):
                buffer += '//'+line+'\n'
            buffer += '//\n'
        self.Xs, self.Ys = zip(*sorted(zip(self.Xs,self.Ys),key=lambda t:t[0]))
        if self.mode == 'normal':
            temp_map:Mapping[int,List[int]] = dict()
            for x,y in zip(self.Xs,self.Ys):
                if x not in temp_map:
                    temp_map[x] = []
                temp_map[x].append(y)
            for x in range(self.shape[0]):
                buffer += ' '.join(map(lambda t:'0' if t not in temp_map.get(x,[]) else '1',range(self.shape[1])))+'\n'
        elif self.mode == 'sparse':
            buffer = f'.sparse:{self.shape[0]}*{self.shape[1]}\n'
            prev_x = None
            for x,y in zip(self.Xs,self.Ys):
                if x != prev_x:
                    buffer += '\n'
                    prev_x = x
                buffer += f'<{x},{y}>'
        if self.compress:
            with open(self.filename,'wb') as file:
                file.write(zlib.compress(buffer.encode()))
        else:
            with open(self.filename,'w') as file:
                file.write(buffer)

def writer(filename:str, cont_matrix:np.ndarray,comments=None,mode='normal',compress=False):
    H,W = cont_matrix.shape
    buffer = ''
    if mode == 'normal':
        for comment in comments:
            buffer += '//'+str(comment)+'\n'
        buffer += f'//H:{H},W:{W}\n'
        for row in cont_matrix:
            buffer += (' '.join(map(lambda x:str(int(x)),row))+'\n')
    elif mode == 'sparse':
        buffer = f'.sparse:{H}*{W}\n'
        if comments:
            for comment in comments:
                buffer += f'//{comment}\n'
        for x in range(H):
            for y in np.nonzero(cont_matrix[x])[0]:
                assert isinstance(x,(int,np.int64)),(x,type(x))
                assert isinstance(y,(int,np.int64)),(y,type(y))
                buffer += f'<{x},{y}>'
            buffer += '\n'
    if compress:
        assert filename.endswith('.zip')
        with open(filename,'wb') as file:
            file.write(zlib.compress(buffer.encode()))
    else:
        with open(filename,'w') as file:
            file.write(buffer)

def _neglect(string_set):
    def f(string:str):
        for i in string_set:
            string = string.replace(i,'')
        return string
    return f
_preprocessor = _neglect(['[',']'])

def _anysum(*to_add):
    if 0 == len(to_add):
        return None
    elif 1 == len(to_add):
        return to_add[0]
    return _anysum(to_add[0]+to_add[1],*to_add[2:])

def _splitby(string_set):
    def f(string:str):
        groups = [string]
        for i in string_set:
            groups = _anysum(*(sub_string.split(i) for sub_string in groups))
        return groups
    return f
_splitter = _splitby((',','\t',' '))

sparse_pattern_finder = re.compile(r'<(\d+),(\d+)>')
sparse_header_finder = re.compile(r'.sparse:(\d+)\*(\d+)')

def read_matrix(filename:str):
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist")
    if filename.endswith('.zip'):
        with open(filename,'rb') as file:
            contents = zlib.decompress(file.read()).decode()
    else:
        with open(filename,'r') as file:
            contents = file.read()
    return _read_matrix(contents)

def _read_matrix(contents:str):
    lines = iter(contents.split('\n'))
    mode = ''
    try:
        line = next(lines)
        while (not line) or line.startswith('//') or line.startswith('.'):
            if line.startswith('.sparse') and not mode:
                h,w = sparse_header_finder.match(line).groups()
                h,w = map(int,(h,w))
                mode = 'sparse'
            elif line.startswith('.diff') and not mode:
                raise NotImplemented
            line = next(lines)
    except StopIteration:
        return None
    if True:
        if mode == 'sparse':
            try:
                board = np.zeros((h,w),dtype='int8')
                while True:
                    if line and not line.startswith('//'):
                        for x,y in sparse_pattern_finder.findall(line):
                            board[int(x),int(y)] = 1
                    line = next(lines)
            except StopIteration:
                return board
        else:
            try:
                rows = []
                while True:
                    if line and not line.startswith('//'):
                        line = _splitter(_preprocessor(line.rstrip()))
                        line = tuple(map(int,line))
                        rows.append(line)
                    line = next(lines)
            except StopIteration:
                return np.array(rows,dtype='int8')



OUTPUT_FORMS = ('COORDS','ARRAY')
def from_coords_to_array(points:Iterable)->np.ndarray:
    board = None
    for x,y in points:
        if not board:
            w,h = points[0]
            board = np.zeros((h,w),dtype='int8')
        else:
            board[x][y] = 1
    return board


def get_reader(output_form='ARRAY'):
    return read_matrix
    assert output_form in OUTPUT_FORMS
    if output_form == 'COORDS':
        return read_matrix
    elif output_form == 'ARRAY':
        def f(*args,**kargs):
            return from_coords_to_array(read_matrix(*args,**kargs))
        return f

if __name__ == "__main__":
    #print(read_matrix("../2*46*199.txt"))
    print(_read_matrix(""".sparse:5*6
<0,0>,<1,1>,<4,5>
<0,2>,<0,4>
"""))
    
    with Writer(5,6,'test.txt',mode='normal',compress=False,comments=['test']) as writer:
        writer[0,0] = 1
        writer[1,1] = 1
        writer[4,5] = 1
        writer[0,2] = 1
        writer[0,4] = 1
    print(read_matrix('test.txt'))
