import cv2
import sys

class Cfg:

    def __init__(self):
        self.sliders = {}
        self.toggles = {}
        self._redraw = None

    def redraw(self, img=None):
        r = self._redraw
        self._redraw = img
        return r


    def get_toggle(self, key, max_, callback):
        key = ord(key)
        if key not in self.toggles:
            self.toggles[key] = {'state': 0, 'has_changed': True, 'callback': callback}

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

