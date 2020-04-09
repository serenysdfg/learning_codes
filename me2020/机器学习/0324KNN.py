def euclidean(v1,v2):
  d=0.0
  for i in range(len(v1)):
    d+=(v1[i]-v2[i])**2
  return math.sqrt(d)

#需要预估的值和数据集的距离
def getdistances(data,vec1):
  distancelist=[]
  
  # Loop over every item in the dataset
  for i in range(len(data)):
    vec2=data[i]['input']
    
    # Add the distance and the index
    distancelist.append((euclidean(vec1,vec2),i))
  
  # Sort by distance
  distancelist.sort()
  return distancelist
#knn估计价格
def knnestimate(data,vec1,k=5):
  # Get sorted distances
  dlist=getdistances(data,vec1)
  avg=0.0
  
  # Take the average of the top k results
  for i in range(k):
    idx=dlist[i][1]
    avg+=data[idx]['result']
  avg=avg/k
  return avg
  #加权knn

def weightedknn(data,vec1,k=5,weightf=gaussian):
  # Get distances
  dlist=getdistances(data,vec1)
  avg=0.0
  totalweight=0.0

  # Get weighted average
  for i in range(k):
    dist=dlist[i][0]
    idx=dlist[i][1]
    weight=weightf(dist)
    avg+=weight*data[idx]['result']
    totalweight+=weight
  if totalweight==0: return 0
  avg=avg/totalweight
  return avg