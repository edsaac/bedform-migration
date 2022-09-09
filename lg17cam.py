from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
from PIL.ExifTags import TAGS
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

## Path to ImageMagick executable
PATH_TO_MAGICK     = r"/home/edsaa/Apps/ImageMagick/magick"

## Camera corrections
BARREL_PARAMS  = (0.000, -0.015, 0.000)
    
## Four-point perspective correction
PERS_ORIGIN_LEFT  = {'x':[0,1200,1200,0],'y':[533,532,99,87]}
PERS_ORIGIN_RIGHT = {'x':[0,1200,1200,0],'y':[495,500,64,47]}
PERS_TARGET_LEFT  = {'x':[0,1200,1200,0],'y':[515,515,75,75]}
PERS_TARGET_RIGHT = {'x':[0,1200,1200,0],'y':[515,515,75,75]}

## Overlaping distance
XSHIFT = 1096 + 18  

## Binarizing parameters
MASKING_THRESHOLD = 100

## Peak-finding parameters
SAVGOL_FILTER_PARAMS = dict(window_length=171,polyorder=4)
TROUGH_FINDER_PARAMS = dict(distance=200,height=15,prominence=10)
PEAK_FINDER_PARAMS   = dict(distance=200,prominence=10)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def fixBarrelDistortion(imgPath,params):
    '''
    Uses imagemagick to fix barrel/pincushion distortion
    in
        str:imgPath  path to image to fix  
        tup:params   three parameters to fix barrel distortion
    out
        str: path to fixed image
    '''
    a,b,c = params
    pathOut = imgPath.replace(".JPG","_barrel.JPG")
    os.system(PATH_TO_MAGICK + " convert \
               {0:} -virtual-pixel black -distort Barrel \
               \"{1:.4f} {2:.4f} {3:.4f}\" {4:}".format(imgPath,a,b,c,pathOut))
    return pathOut

def fixPerspective(imgPath,pointsFrom,pointsTarget):
    '''
    Uses imagemagick to get a 4-point perspective distortion
    in
        str:imgPath        path to image to fix  
        dict:pointsFrom    {'x':[x1,x2,x3,x4],'y':[x1,x2,x3,x4]}
        dict:pointsTarget  {'x':[x1,x2,x3,x4],'y':[x1,x2,x3,x4]}
    out
        str: path to fixed image
    '''
    originPairs = list(zip(pointsFrom['x'],pointsFrom['y']))
    targetPairs = list(zip(pointsTarget['x'],pointsTarget['y']))
    
    pathOut = imgPath.replace(".JPG","_perspective.JPG")
    
    listCoords = ""
    for o,t in zip(originPairs,targetPairs):
        listCoords += f"{o[0]},{o[1]} {t[0]},{t[1]}  "
    
    os.system(PATH_TO_MAGICK + " convert \
               {0:} -matte -virtual-pixel black -distort Perspective \
               '{1:}' {2:}".format(imgPath,listCoords,pathOut))
    return pathOut

def checkTimeStamps(timeStamps):
    timefmt = '%Y:%m:%d %H:%M:%S'
    dates = list(timeStamps.values())
    t1 = pd.to_datetime(dates[0],format=timefmt)
    t2 = pd.to_datetime(dates[-1],format=timefmt)
    deltat = abs(t1-t2)

    if deltat > pd.to_timedelta("00:00:30"):
        print("Looks like the timestamps do not coincide!")
        print(f"  Left: {t1} \n  Right:{t2}")

def processPair(leftPath,rightPath,result_folders):
    raw_paths = {'left':leftPath, 'right':rightPath}
    raw_imgs  = {k:Image.open(v) for k,v in raw_paths.items()}
    raw_imgs_timestamps = {k:v.getexif().get(306) for k,v in raw_imgs.items()}
    checkTimeStamps(raw_imgs_timestamps)

    # Fix barrel distortion
    barr_paths = {k:fixBarrelDistortion(v,BARREL_PARAMS) for k,v in raw_paths.items()} 
    barr_imgs  = {k:Image.open(v) for k,v in barr_paths.items()}
    
    # Fix perspective distortion
    pers_paths = dict()
    pers_paths['left']  = fixPerspective(barr_paths['left'],PERS_ORIGIN_LEFT,PERS_TARGET_LEFT)
    pers_paths['right'] = fixPerspective(barr_paths['right'],PERS_ORIGIN_RIGHT,PERS_TARGET_RIGHT)
    pers_imgs  = {k:Image.open(v) for k,v in pers_paths.items()}
    
    # Crop to conserve only water/sediment
    crop_imgs  = {k:v.crop((50, 250, v.size[0], 500)) for k,v in pers_imgs.items()}

    # Normalize both pictures
    np_imgs   = {k:np.asarray(v.convert('L')) for k,v in crop_imgs.items()}
    equa_imgs = {k:np.interp(v, (v.min(), v.max()), (0, 255)) for k,v in np_imgs.items()}
    inte_imgs = {k:Image.fromarray(v.astype(np.uint8),"L") for k,v in equa_imgs.items()}

    # Overlap images
    (width1, height1) = inte_imgs['left'].size
    (width2, height2) = inte_imgs['right'].size

    result_width = width1 + XSHIFT
    result_height = max(height1, height2)

    joined_img = Image.new('L', (result_width, result_height))
    joined_img.paste(im=inte_imgs['right'], box=(XSHIFT, 0))
    joined_img.paste(im=inte_imgs['left'], box=(0, 0))
         
    # Color classification for edge detection
    hits = joined_img.histogram()
    masked = np.ma.masked_greater(joined_img,MASKING_THRESHOLD)
    ycoord = np.ma.count_masked(masked,axis=0)

    # Filter and smooth the edge profile
    ysmoothed = savgol_filter(ycoord,**SAVGOL_FILTER_PARAMS)

    # Locate peaks and throughs
    whereTroughs,_ = find_peaks(ysmoothed,**TROUGH_FINDER_PARAMS)
    wherePeaks,_   = find_peaks(-ysmoothed,**PEAK_FINDER_PARAMS)

    # Save the extracted X-data
    timestamp = str(raw_imgs_timestamps['left'])
    troughs_Xpx = ','.join(str(w) for w in whereTroughs)
    peaks_Xpx   = ','.join(str(w) for w in wherePeaks)

    troughs_Ypx = ','.join("{:.1f}".format(ysmoothed[w]) for w in whereTroughs)
    peaks_Ypx   = ','.join("{:.1f}".format(ysmoothed[w]) for w in wherePeaks)

    path_file_troughs = os.path.join(result_folders['PATH_FOLDER_TROUGHS'],raw_imgs_timestamps['left'])
    path_file_peaks   = os.path.join(result_folders['PATH_FOLDER_PEAKS'],raw_imgs_timestamps['left'])

    with open(path_file_troughs,'w') as f:
        f.write(f"[{timestamp}]\t[{troughs_Xpx}]\t[{troughs_Ypx}]\n")

    with open(path_file_peaks,'w') as f:
        f.write(f"[{timestamp}]\t[{peaks_Xpx}]\t[{peaks_Ypx}]\n")

    ## Create Line over image
    endimg = Image.new('RGB', joined_img.size)
    endimg.paste(im=joined_img, box=(0, 0))

    lineimg = ImageDraw.Draw(endimg)
    lineimg.line(list(zip(np.arange(joined_img.size[0]),ysmoothed)),
                 fill ="orange", width = 7)

    ## Create Points over image
    r = 10 
    for wt in whereTroughs:
        lineimg.ellipse([(wt-r,ysmoothed[wt]-r),(wt+r,ysmoothed[wt]+r)],
                        outline = 'white', fill = 'red', width = 1)

    for wt in wherePeaks:
        lineimg.ellipse([(wt-r,ysmoothed[wt]-r),(wt+r,ysmoothed[wt]+r)], 
                        outline = 'white', fill = 'blue', width = 1)
    # Save results
    path_file_endimg = os.path.join(result_folders['PATH_FOLDER_ENDIMG'],raw_imgs_timestamps['left'])
    endimg.save(path_file_endimg+".JPG")

    # Clean temporal files
    for f in barr_paths.values(): os.remove(f) 
    for f in pers_paths.values(): os.remove(f)
    
    return 0

def getDatabasePics(dirPath):

    listImg = os.listdir(dirPath)
    dateImg = []
    
    for f in listImg:
        path = os.path.join(dirPath,f)
        img = Image.open(path)
        exifdata = img.getexif()
        
        tag = TAGS.get(306)
        date = exifdata.get(306)
        if isinstance(date, bytes): date = date.decode()
        dateImg.append(date)
        
    db = pd.DataFrame({'File':listImg, 'Time':dateImg})
    db['Time'] = pd.to_datetime(db['Time'],format='%Y:%m:%d %H:%M:%S')
    db.sort_values(by=['Time'],inplace=True,ignore_index=True)
    return db 

def catFolder(listInput,fileOutput):
    import fileinput
    with open(fileOutput, 'w') as fout, fileinput .input(listInput) as fin:
        for line in fin:
            fout.write(line)
    return None

def readProcessedPairs(filename,sep='\t'):
    timefmt = '%Y:%m:%d %H:%M:%S'
    
    with open(filename) as f:
        lines = f.readlines()
        time = [line.split(sep)[0].replace('\n','').replace('[','').replace(']','') for line in lines]
        xpx  = [line.split(sep)[1].replace('\n','').replace('[','').replace(']','').split(',') for line in lines]
        ypx  = [line.split(sep)[2].replace('\n','').replace('[','').replace(']','').split(',') for line in lines]
    
    Index,Tvec,Xvec,Yvec = [],[],[],[]
    i = 0
    
    for t,X,Y in zip(time,xpx,ypx):
        for x,y in zip(X,Y):
            Index.append(i)
            Tvec.append(pd.to_datetime(t,format=timefmt))
            Xvec.append(float(x))
            Yvec.append(float(y))
        i += 1
    
    df = pd.DataFrame({'I':Index,'Time':Tvec,'X(px)':Xvec,'Y(px)':Yvec})
    return df