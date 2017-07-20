import cv2
import sys
import numpy as np

class Context:

    def __init__(self):
        self.sliders = {}
        self.toggles = {}
        self._redraw = False
        self.cur_buf_id = 0;
        self.buffers = []
        self.buffers_by_name = {}
        self._once = []
        cv2.namedWindow('image')

    def once(self, key):
        if key in self._once:
            return False
        self._once.append(key)
        return True

    def redraw(self, *_):
        self._redraw = True

    def add_buffer(self, name, shape=[], src=None):
        if name not in self.buffers_by_name:
            img = src if src is not None else np.zeros(shape, np.uint8)
            self.buffers.append(img)
            self.cur_buf_id = len(self.buffers)-1
            self.buffers_by_name[name] = (self.buffers[-1], self.cur_buf_id)

    def b(self, name):
        return self.buffers_by_name[name][0]

    def __setitem__(self, key, value):
        if key in self.buffers_by_name:
            id = self.buffers_by_name[key][1]
            self.buffers_by_name[key] = (value, id)
            self.buffers[id] = value
        else:
            self.add_buffer(key, src=value)


    def __getitem__(self, key):
        if key in self.buffers_by_name:
            return self.buffers_by_name[key][0]
        return None


    def get_toggle(self, key, max_, callback, init=0):
        key = ord(key)
        if key not in self.toggles:
            self.toggles[key] = {'state': init, 'has_changed': True, 'callback': callback}

        ko = self.toggles[key]
        ko['callback'] = callback
        has_changed = ko['has_changed']
        ko['has_changed'] = False
        if ko['state'] > max_:
            ko['state'] = 0

        return (ko['state'], has_changed)

    def got_key(self, key):
        if key in self.toggles:
            ko = self.toggles[key]
            ko['state'] += 1
            ko['has_changed'] = True
            if ko['callback'] is not None:
                ko['callback'](None)
            print "Key:", chr(key), "=", ko['state']
            sys.stdout.flush()

        (_, ffd ) = self.get_toggle('b', 1, None, init=0)
        (_, back) = self.get_toggle('v', 1, None, init=0)
        if back:
            self.cur_buf_id -= 1
            self._redraw = True
        if ffd:
            self.cur_buf_id += 1
            self._redraw = True
        self.cur_buf_id = self.cur_buf_id % len(self.buffers)

    def get_slider(self, name, callback=None, init=0, max_=255):
        created = False
        if name not in self.sliders:
            def none():
                pass
            if callback == None:
                callback = none
            self.sliders[name] = {'old_value': init}
            cv2.createTrackbar(name,'image',init,max_,callback)
            created = True

        val = cv2.getTrackbarPos(name,'image')
        old_val = self.sliders[name]['old_value']
        self.sliders[name]['old_value'] = val
        return (val, val != old_val or created)


    def eventloop(self):

        while(1):
            k = cv2.waitKey(100) & 0xFF

            if k != 255:
                self.got_key(k)
            if k == 27 or k == ord('q'):
                break

            if self._redraw:
                self._redraw = False
                print "imshow"
                sys.stdout.flush()
                cv2.imshow('image', self.buffers[self.cur_buf_id])

        cv2.destroyAllWindows()
