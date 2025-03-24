class xrange:
    def __init__(self,*bounds):
        if len(bounds) == 0:
            assert False,"Empty Input!"
        if len(bounds) == 1:
            bounds = bounds[0]
        assert isinstance(bounds,(list,tuple)),f"boundary should be list or tuple, get {bound} instead!"
        #assert bounds[0] < bounds[1] , f"lower_bound {bounds[0]} should be smaller than upper_bound {bounds[1]}"
        self._lower = bounds[0]
        self._upper = bounds[1]
        self._step = 0
        if len(bounds) == 3:
            self._step = bounds[2]
            pass
            #assert self._lower + self._step <= self._upper,f"step too large"
        self._pos = -1
            
    def size(self):
        return self._upper - self._lower

    def __add__(self,offset):
        return xrange(self._lower+offset,self._upper+offset,self._step)
    
    def __sub__(self,offset):
        return xrange(self._lower-offset,self._upper-offset,self._step)
    
    def __mul__(self,mult):
        return xrange(self._lower*mult,self._upper*mult,self._step*mult)
    
    def __truediv__(self,divi):
        return xrange(self._lower/divi,self._upper/divi,self._step/divi)

    def upper_bound(self):
        return self._upper
    
    def lower_bound(self):
        return self._lower
    
    def step(self):
        return self._step
    
    def __len__(self):
        return 1+int(self.size()//self._step)
    
    def __contains__(self,val):
        if not self._step:return self._lower<=val<=self._upper
        return self._lower<=val<=self._upper
    
    def __iter__(self):
        assert self._step, "Not iterable"
        self._pos = 0
        return self
    
    def __next__(self):
        self._pos += 1
        if self._pos > self.__len__():raise StopIteration()
        return self._lower+(self._pos-1)*self._step
