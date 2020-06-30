#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Use FFMPEG to create a movie of all files in a folder.
'''

# General imports
import subprocess
from pathlib import Path
import logging

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_movie(fpath_in, fps=None, fpath_out=None):
    ''' Create a movie of image files in a folder.

    Parameters
    ----------
    fps :       int
                output fps
    fpath_in :  str
                filename of an image, or
                foldername containing images (looking for .png)
    fpath_out : str
                name of outputfile,
                'foldername.avi' is default

    '''
    fps_out = 25
    fpath_in = Path(fpath_in).resolve()

    if fps is None:
        fps = 5

    if fpath_in.is_file():
        suffix = fpath_in.suffix
        fpath_in = fpath_in.parent
    else:
        suffix = '.png'

    if fpath_out is None:
        fpath_out = Path(fpath_in, str(fpath_in.name) + '.avi')
    else:
        fpath_out = Path(fpath_out)

    fpath_in = str(fpath_in)
    fpath_out = str(fpath_out)

    # - framerate --> framerate in
    # - r         --> framerate out
    # ffmpeg -f image2 -framerate 5 -i %*.png -r 25 out.avi
    # command = ('ffmpeg -y -f image2 -framerate {} -i {}/%*{} -r {} {}'
    #            ''.format(fps, fpath_in, suffix, fps_out, fpath_out)
    #            )
    command = ('ffmpeg -framerate {} -i {}/%*{} -r {} -qscale:v 5 {}'
               ''.format(fps, fpath_in, suffix, fps_out, fpath_out)
               )

    # media 9
    # ffmpeg -i movie.avi -sameq -vcodec libx264 -x264opts keyint=25 movie.mp4
    # or
    # ffmpeg -i movie.avi -sameq  movie.flv

    # idea:
    # ImageMagik
    # convert -verbose -delay 0.01 -loop 1 fig*.png animation.gif

    logger.info('Running command:\n' + command + '\n')
    subprocess.call(command, shell=True)


#
