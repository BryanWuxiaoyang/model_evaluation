import json
import os
from utils import *


class AutoSave:
    """
    Helping deal with problems and issues in process of processing large-scale datasets, by automatically saving and
    loading data from-and-to designated files.

    The instances of this class are often used in conjunction with [AutoSaveIterator] to implement breakpoint-resuming.

    The whole data is partitioned into two parts, respectively [disk] and [memory], with property being:
    [disk] union [memory] == [complete_data]
    There's no requirement that [disk] intersect [memory] == {}
    By [load] method, disk data shall be merged into memory.
    By [save] method, memory data shall be merged into disk
    By [clear] method, memory data shall be merged into disk, and memory data shall be cleared.

    Sub classes should implement [_load], [_save], [_clear],
    """
    def __init__(self, file_path: str, lazy_load: bool = False, print_info: bool = False):
        """
        :param file_path:
        :param lazy_load: Load data only when
        :param print_info: Printing information when enforcing saving or loading actions
        """
        self.file_path = file_path
        self.lazy_load = lazy_load
        self.print_info = print_info
        self.loaded = False

        if not self.lazy_load:
            self.load()

    def _load(self):
        raise NotImplemented

    def load(self):
        if not self.loaded:
            if self.print_info: print("loading")
            self._load()
            self.loaded = True
            if self.print_info: print("loaded")

    def _save(self):
        raise NotImplemented

    def save(self):
        self.load()
        if self.print_info: print("saving")
        self._save()
        if self.print_info: print("saved")

    def _clear(self):
        raise NotImplemented

    def clear(self):
        self.load()
        self.save()
        self._clear()
        self.loaded = False


class AutoSaveList(AutoSave):
    def __init__(self, file_path: str, lazy_load: bool = False, print_info: bool = False):
        self.data = []
        super(AutoSaveList, self).__init__(file_path, lazy_load, print_info)

    def _load(self):
        if not os.path.exists(self.file_path): return
        with open(self.file_path, "r", encoding="utf8") as f:
            data = json.load(f)
            data += self.data
            self.data = data

    def _save(self):
        if len(self.data) == 0: return
        with open(self.file_path, "w", encoding="utf8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def _clear(self):
        self.data.clear()

    def __add__(self, lst):
        for element in lst:
            self.append(element)

    def __iter__(self):
        self.load()
        return self.data.__iter__()

    def append(self, element):
        self.data.append(element)

    def __getitem__(self, item: int):
        self.load()
        return self.data[item]


class AutoSaveDict(AutoSave):
    def __init__(self, file_path: str, lazy_load: bool = False, print_info: bool = False):
        super(AutoSaveDict, self).__init__(file_path, lazy_load, print_info)
        self.data = {}

    def _load(self):
        if not os.path.exists(self.file_path): return
        with open(self.file_path, "r", encoding="utf8") as f:
            data = json.load(f)
            if len(self.data) >= len(data):
                self.data.update(data)
            else:
                data.update(self.data)
                self.data = data

    def _save(self):
        with open(self.file_path, "w", encoding="utf8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def _clear(self):
        self.data.clear()

    def __getitem__(self, item):
        self.load()
        return self.data.get(item)

    def put(self, key, value):
        self.data[key] = value

    def __iter__(self):
        self.load()
        return self.data.__iter__()


class AutoSaveIterator:
    """
    The iterator that could auto-save data at a designated rate while fetching data, and load data through breakpoint-resuming.
    """
    def __init__(self, iterator_generator, resume_iter_file_path: str, loader, saver,
                 save_every: int, auto_save: bool = True, auto_iter: bool = True, restart: bool = False,
                 print_info: bool = False):
        """
        :param iterator_generator: Raw iterator to fetch element from.
        :param resume_iter_file_path: The file used to store breakpoint iteration information.
        :param loader: (iter_cnt->None), The function used to load necessary data.
        :param saver: (iter_cnt->None), The function used to save necessary data.
        :param save_every: The number of iterations between each save action.
        :param auto_save: Whether [saver] should be automatically called when savable.
        If set to [False], user should call [try_save] method to try to save data when savable.
        :param auto_iter: Whether [iter_cnt] should be refreshed each time __next__ is called.
        If set to [False], [iter_cnt] would not be incremented for each __next__, therefore user should call [try_iter]
        method to increase [iter_cnt] and do save actions when available.

        Note: set it to False if processing data in parallel, and call [try_iter] in a single thread,
        or the [resume_iter_file] may record iteration counter that doesn't match the real data.

        :param print_info: Printing prompting information when saving or loading
        """
        self.iterator_generator = iterator_generator
        self.resume_iter_file_path = resume_iter_file_path
        self.loader = loader
        self.saver = saver
        self.save_every = save_every
        self.auto_save = auto_save
        self.auto_iter = auto_iter
        self.print_info = print_info

        if restart:
            self.current_iter = 0
            self.save_iter()
        else:
            self.load_iter()

        self.load()
        self.iterator = iterator_generator(self.current_iter)
        self.save_flag = False

    def __next__(self):
        try:
            ret = self.iterator.__next__()
            if self.auto_iter:
                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.save_flag = True
                    if self.auto_save:
                        self.try_save()
            return ret
        except StopIteration:
            self.save_flag = True
            if self.auto_save:
                self.try_save()
            raise StopIteration

    def __iter__(self):
        return self

    def load_iter(self):
        self.current_iter = int(load_value(self.resume_iter_file_path, default=0))

    def save_iter(self):
        save_value(self.resume_iter_file_path, value=str(self.current_iter))

    def load(self):
        if self.loader is not None:
            self.loader(self.current_iter)

    def save(self):
        if self.print_info: print("saving_iterator: ", self.current_iter)

        if self.saver is not None: self.saver(self.current_iter)
        self.save_iter()

        if self.print_info: print("saved_iterator: ", self.current_iter)

    def try_save(self):
        if self.save_flag:
            self.save()
            self.save_flag = False

    def try_iter(self):
        self.current_iter += 1
        if self.current_iter % self.save_every == 0:
            self.save_flag = True
            if self.auto_save:
                self.try_save()


class AutoSaveIterable:
    def __init__(self, iterable_generator, resume_iter_file_path: str, loader, saver, save_every: int, auto_save: bool = True, restart: bool = False,
                 print_info: bool = False):
        self.iterable_generator = iterable_generator
        self.resume_iter_file_path = resume_iter_file_path
        self.loader = loader
        self.saver = saver
        self.save_every = save_every
        self.auto_save = auto_save
        self.restart = restart
        self.print_info = print_info

        self.iterator_generator = lambda iter_cnt: self.iterable_generator(iter_cnt).__iter__()

    def __iter__(self):
        return AutoSaveIterator(self.iterator_generator, self.resume_iter_file_path, self.loader, self.saver,
                                self.save_every, self.auto_save, self.restart, self.print_info)


if __name__ == "__main__":
    l = AutoSaveList(file_path="test", lazy_load=False, print_info=True)
    l.append(1)

    pass




