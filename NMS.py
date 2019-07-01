# python3
import numpy as np

def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
 #图片的四条边
    x1=dets[:,0]
    y1=dets[:,1]
    x2=dets[:,2]
    y2=dets[:,3]
  #置信度得分
    score=dets[:,4]
 #将置信度得分按降序排列
    order=score.argsort()[::-1]
  #图片面积
    areas=(x2-x1+1)*(y2-y1+1)

    keep=[]
    while order.size>0:
        i=order[0]
        keep.append(i)
 #重叠区域的的面积
        x11=np.maximum(x1[i],x1[order[1:]])
        y11=np.maximum(y1[i],y1[order[1:]])
        x12=np.minimum(x2[i],x2[order[1:]])
        y12=np.minimum(y2[i],y2[order[1:]])
 #计算面积，负值的话取0
        w=np.maximum(0.0,x12-x11+1)
        h=np.maximum(0.0,y12-y11+1)

        inters=w*h
 #计算其
        iou=inters/(areas[i]+areas[order[1:]]-inters)
 #大于阈值的iou都不要
        ids=np.where(iou<=thresh)[0]
 #按照顺序再次排列
        order=order[ids+1]
    return keep
# tesy
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1],
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])
    thresh = 0.35
    keep_dets = py_nms(dets, thresh)
    print(keep_dets)
    print(dets[keep_dets])

