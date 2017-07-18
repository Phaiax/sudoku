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
        cv2.namedWindow('image')

    def redraw(self, redraw=True):
        (self.cur_buf_id, _) = self.get_toggle('b',
            len(self.buffers)-1,
            self.redraw,
            init=self.cur_buf_id)
        r = self.buffers[self.cur_buf_id] if self._redraw else None
        self._redraw = redraw
        return r

    def add_buffer(self, name, shape=[], src=None):
        if name not in self.buffers_by_name:
            img = src if src is not None else np.zeros(shape, np.uint8)
            self.buffers.append(img)
            self.buffers_by_name[name] = self.buffers[-1]
            self.cur_buf_id = len(self.buffers)-1

    def b(self, name):
        return self.buffers_by_name[name]

    def get_toggle(self, key, max_, callback, init=0):
        key = ord(key)
        if key not in self.toggles:
            self.toggles[key] = {'state': init, 'has_changed': True, 'callback': callback}

        ko = self.toggles[key]
        ko['callback'] = callback
        if ko['state'] > max_:
            ko['state'] = 0

        return (ko['state'], ko['has_changed'])

    def got_key(self, key):
        if key in self.toggles:
            ko = self.toggles[key]
            ko['state'] += 1
            ko['has_changed'] = True
            ko['callback'](None)
            print "Key:", chr(key), "=", ko['state']

    def get_slider(self, name, callback=None, min_=0, max_=255):
        if name not in self.sliders:
            def none():
                pass
            if callback == None:
                callback = none
            self.sliders[name] = {'old_value': min_}
            cv2.createTrackbar(name,'image',min_,max_,callback)


        val = cv2.getTrackbarPos(name,'image')
        old_val = self.sliders[name]['old_value']
        self.sliders[name]['old_value'] = val

        return (val, val != old_val)


    def eventloop(self):

        while(1):
            k = cv2.waitKey(1) & 0xFF

            self.got_key(k)
            if k == 27 or k == ord('q'):
                break

            img = self.redraw()
            if img is not None:
                cv2.imshow('image',img)

        cv2.destroyAllWindows()
