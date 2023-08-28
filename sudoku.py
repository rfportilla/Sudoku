#Classes for sudoku game
from typing import Iterable
from itertools import combinations
from pickle import FALSE


class Notes(set):
    def __add__(self, other):
        return self.union(set(iter(other)))
            
        


class Cell(object):
    _value = None
    _tvalue = None
    _notes: Notes = None
    _min = 0
    _max = 10 ** 3
    x = y = None  # optional position in grid

    def __init__(self, value=0, x=None, y=None, maximum=None):
        self._value = value
        self._tvalue = 0
        self.notes = Notes()
        self.x = x
        self.y = y
        if maximum is not None and maximum > 0:
            self._max = maximum

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v > self._max or v < self._min:
            raise CellValueError('Value is out of range.  Must be between {} and {}'.format(self._min, self._max))
        self._value = v
 
    def reset_tvalue(self):
        self.tvalue = 0

    def set_tvalue(self):
        self.value = self.tvalue

    @property
    def tvalue(self):
        return self.value if self.value != 0 else self._tvalue

    @tvalue.setter
    def tvalue(self, v):
        if self.value != 0 and v != 0:
            raise SetTestValueError('Cannot set non-zero test value when cell is already solved.')
        if v > self._max or v < self._min:
            raise CellValueError('Note is out of range.  Must be between {} and {}'.format(self._min, self._max))
        self._tvalue = v

    @property
    def notes(self):  # only show notes if value is 0 or None
        return self._notes if self.value == 0 or self.value is None else set()

    @notes.setter
    def notes(self, s):
        s = Notes(s)
        if len(s) > 0 and (max(s) > self._max or min(s) < self._min):
            raise CellValueError('Note is out of range.  Must be between {} and {}'.format(self._min, self._max))

        self._notes = s

    def add_note(self, n: int):
        if n > self._max or n < self._min:
            raise CellValueError('Note is out of range.  Must be between {} and {}'.format(self._min, self._max))
        self.notes.add(n)

    def del_note(self, n: int):
        try:
            self.notes.remove(n)
        except KeyError:
            raise ValueNotPresent('Value \'{}\' not in notes.'.format(n))

    def eval(self):
        if self.value == 0 and len(self.notes) == 1:
            self.value = list(self.notes)[0]

    @property
    def pos(self):
        return (self.x, self.y)

    def __repr__(self):
        return repr(self.value)
    
    def __int__(self):
        return int(self.value)

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __hash__(self):
        return self.value



class ValueNotPresent(Exception):
    pass


class CellValueError(Exception):
    pass

class SetTestValueError(Exception):
    pass

class Sudoku(list):
    dimension = 9

    def __init__(self, l=None):
        super().__init__(list() if l is None else l)
        self.dimension = self.dimension if l is None else len(l)
        self._create_cells()

        if l is not None:
            for y in range(self.dimension):
                for x in range(self.dimension):
                    self[y][x] = Cell(l[y][x], x=x, y=y, maximum=self.dimension)


    def _create_cells(self):
        while len(self) < self.dimension:
            self.append([])
        for y in range(self.dimension):
            while len(self[y]) < self.dimension:
                self[y].append(Cell(0, x=len(self[y]), y=y, maximum=self.dimension))
        
    @classmethod
    def from_vector(cls, prefill=None, dimension=9):
        o = cls()
        o.dimension = dimension

        if prefill is not None:
            for vector in prefill:
                o[vector[0]][vector[1]] = Cell(vector[2], x=vector[0], y=vector[1], maximum=o.dimension)
        return o


    def cells(self):
        for y in range(self.dimension):
            for x in range(self.dimension):
                yield self[y][x]

    def reset_test_values(self):
        for cell in self.cells():
            cell.reset_tvalue()

    def set_test_values(self):
        for cell in self.cells():
            cell.set_tvalue()

    def is_valid(self):
        for i in range(self.dimension):
            m = int(i/3)
            n = i - m * 3

            r = [x for x in self.row(i) if x != 0]
            c = [x for x in self.column(i) if x != 0]
            b = [x for x in self.block(m, n) if x != 0]
            for x in [r, c, b]:
                if len(x) != len(set(x)):
                    return False
        return True

    def is_solved(self):
        self.is_valid()
        for cell in self.cells():
            if cell.value == 0:
                return False
        return True
                
    def is_tsolved(self):
        for cell in self.cells():
            if cell.tvalue == 0:
                return False
        return True

    def block_from_xy(self, x, y):
        #return (int(x / 3), int(y / 3))
        return (int(x * 3 / (self.dimension)), int(y * 3/ (self.dimension)))

    def block(self, m: int, n: int):
        third = int(self.dimension / 3)
        block = self.create_grid(self.dimension, 1)[0]

        x0 = third * (m+1) - 3
        y0 = third * (n+1) - 3

        for y in range(third):
            for x in range(third):
                block[y*third+x] = self[y+y0][x+x0]
        return block

    def column(self, x):
        column = self.create_grid(self.dimension, 1)[0]
        for y in range(self.dimension):
            column[y] = self[y][x]
        return column
        
    def row(self, y):
        row = self.create_grid(self.dimension, 1)[0]
        for x in range(self.dimension):
            row[x] = self[y][x]
        return row

    @staticmethod
    def create_grid(w, h):
        return [[Cell() for _ in range(w)] for _ in range(h)]

    def __repr__(self):
        string = ''
        v = '|'
        h = '-'
        b1 = int(self.dimension / 3)
        b2 = int(self.dimension * 2 / 3)
        for y in range(self.dimension):
            row = self[y][0:b1] + [v] + self[y][b1: b2] + [v] + self[y][b2:]
            string += ' '.join(str(r) for r in row) + '\n'
            if y + 1 == b1 or y + 1 == b2:
                string += h * (2 * self.dimension + 3) + '\n'
        return string




class SudokuSolver(object):
    game: Sudoku = None


    def __init__(self, game: Sudoku):
        self.game = game

    def fill_notes(self, find_linear_sets:bool=False):
        for cell in self.game.cells():
            values = set()

            for bcell in self.game.block(*self.game.block_from_xy(cell.x, cell.y)):
                values.add(bcell.value)
            for ccell in self.game.column(cell.x):
                values.add(ccell.value)
            for rcell in self.game.row(cell.y):
                values.add(rcell.value)

            cell.notes = set(range(1, self.game.dimension+1)).difference(values)
        if find_linear_sets:
            self.eval_linear_sets()

    def _auto_solve(self, iterations=100):
        for i in range(iterations):
            changes = 0
            if i > 0:
                self.eval_linear_sets()
            changes += self.eval_cross_values()
            changes += self.eval_excludes()

            if changes == 0:
                break

    def _auto_test_values(self, depth = 1, iterations=100):
        if depth > iterations:
            raise RecursionError('Reached max depth of {}'.format(depth))

        tcell = None
        for cell in self.game.cells():
            if cell.tvalue == 0:
                tcell = cell
                break
        else:
            if self.game.is_tsolved():
                self.game.set_test_values()
                return True
            return False

        for n in tcell.notes:
            if (n in [x.tvalue for x in self.game.row(tcell.y)]  # Check if this conflict with existing
                or n in [x.tvalue for x in self.game.column(tcell.x)]
                or n in [x.tvalue for x in self.game.block(*self.game.block_from_xy(*tcell.pos))]):
                continue
            tcell.tvalue = n
            if self._auto_test_values(depth + 1):
                return True
        tcell.reset_tvalue()
        return False


    def solve(self):
        self._auto_solve()
        self.game.reset_test_values()
        self._auto_test_values()
        return self


    def eval_cross_values(self):
        '''
        Solve cells that have only 1 possible option; i.e. len(notes) == 1
        '''
        change_cnt = 0
        for cell in self.game.cells():
            if cell == 0:
                cell.eval()
                if cell != 0:
                    change_cnt += 1
        return change_cnt

    def _exclude(self, group):
        '''
        For each possible value in the group, check if there is exactly one possible cell for that value.
        '''
        change_cnt = 0
        poss_values = set(range(1, self.game.dimension+1)).difference(group)
        positions = {x:[] for x in poss_values}
        for i in range(len(group)):
            cell = group[i]
            for n in poss_values:
                if n in cell.notes:
                    positions[n].append(i)
        for val, pos in positions.items():
            if len(pos) == 1:
                group[pos[0]].value = val
                change_cnt += 1
        return change_cnt

    def eval_excludes(self):
        '''
        Solve cells where it is the only possible cell for a given value.
        '''
        change_cnt = 0

        for i in range(self.game.dimension):
            change_cnt += self._exclude(self.game.column(i))
            change_cnt += self._exclude(self.game.row(i))
            m = int(i/3)
            n = i - m * 3
            change_cnt += self._exclude(self.game.block(m, n))
            self.fill_notes(find_linear_sets=True)
        return(change_cnt)

    def _find_sets(self, group):
        unknown_grp = []
        for cell in group:
            if cell == 0:
                unknown_grp.append(cell)        
        if len(unknown_grp) < 3:
            return 0

        change_cnt = 0
        for n in range(2, len(unknown_grp)-1):
            combos = combinations(range(len(unknown_grp)), r=n)
            for combo in combos:
                poss_values = set()
                for i in range(n):
                    poss_values = poss_values.union(unknown_grp[combo[i]].notes)
                if len(poss_values) == n:
                    for cpos in [x for x in range(len(unknown_grp)) if x not in combo]:
                        cell = unknown_grp[cpos]
                        for remove_note in poss_values:
                            if remove_note in cell.notes:
                                cell.del_note(remove_note)
                        change_cnt += 1
        return change_cnt
                        

    def eval_linear_sets(self):
        '''
        Look for sets of cells that have a limited set of values and update notes
        '''
        change_cnt = 0
        for i in range(self.game.dimension):
            change_cnt += self._find_sets(self.game.column(i))
            change_cnt += self._find_sets(self.game.row(i))
            m = int(i/3)
            n = i - m * 3
            change_cnt += self._find_sets(self.game.block(m, n))
        return change_cnt

    def __repr__(self):
        b1 = int(self.game.dimension / 3) - 1
        b2 = int(self.game.dimension * 2 / 3) - 1
        text = '_' * 39 + '\n'
        for y in range(self.game.dimension):
            lines = ['|', '|', '|']
            for x in range(self.game.dimension):
                c = self.game[y][x]
                if c.value != 0:
                    lines[0] += '/-\\'
                    lines[1] += '|{}|'.format(c.value)
                    lines[2] += '\\-/'
                else:
                    for i in range(self.game.dimension):
                        m = int(i/3)
                        lines[m] += str(i+1) if i+1 in c.notes else ' '
                lines[0] += '||' if x in (b1, b2) else '|'
                lines[1] += '||' if x in (b1, b2) else '|'
                lines[2] += '||' if x in (b1, b2) else '|'
            text += '{}\n{}\n{}\n'.format(lines[0], lines[1], lines[2])
            if y in (b1, b2):
                text += '=' * 39 + '\n'
            else:
                text += '-' * 39 + '\n'
        return(text)



    




if __name__ == '__main__':
    sample1 = [[0, 4, 5, 1, 0, 0, 7, 0, 0], [9, 0, 0, 0, 0, 0, 0, 0, 1], [6, 0, 0, 0, 3, 8, 0, 0, 0], [2, 0, 0, 0, 6, 0, 8, 0, 0], [0, 8, 9, 3, 0, 7, 6, 1, 0], [0, 0, 4, 0, 1, 0, 0, 0, 3], [0, 0, 0, 7, 5, 0, 0, 0, 6], [1, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 7, 0, 0, 3, 1, 9, 0]]

    o = Sudoku(sample1)
    print(o)

    s = SudokuSolver(o)
    print('Is solved? ', s.game.is_solved())

    s.fill_notes()
    print(s)

    s.solve()
    print('Is solved? ', s.game.is_solved())

    print(s)
