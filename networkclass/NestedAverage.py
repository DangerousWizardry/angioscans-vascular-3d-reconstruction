import copy


class NestedAverage:
    def __init__(self,numbers=list(),limit=10,average=0,count=0):
        self.limit = limit
        self.average = average
        self.count = count
        self.numbers = numbers
        if self.average == 0 and self.count == 0:
            self.numbers = []
    
    def add_number(self, number):
        self.numbers.append(number)
        if self.count > self.limit:
            self.numbers.pop(0)
        else:
            self.count += 1
        self.average = sum(self.numbers)/self.count

    def get_average(self):
        return self.average
    
    def get_last(self):
        return self.numbers[-1]
    
    def copy(self):
        return copy.deepcopy(self)