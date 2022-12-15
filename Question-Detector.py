import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.cluster import DBSCAN



rsm = '4.png'
image = cv2.imread(rsm)
imagereel = cv2.imread(rsm)

bol1 = 876
bolorta = 40
topmargin = 140
bottommargin = 144
image3 = image[:,0:bol1]

image1 = image[topmargin:-1*bottommargin, :bol1]
image2 = image[topmargin:-1*bottommargin, bol1+bolorta:]

kontur = np.array([])
kontur2 = np.array([])
kontur_1 = np.array([])
kontur_1_2 = np.array([])
kontur_2 = np.array([])
kontur_2_2 = np.array([])


for bol in range(1,3):
    if (bol==1):
        image=image1
    else:
        image=image2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Morph operations
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel, iterations=0)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    #dilate = cv2.dilate(opening, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    dilate = cv2.dilate(opening, kernel, iterations=1)
    
    # Remove center line
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h)
        # if (area > 10000 and area < 12500 and ar < .5):
        #     cv2.drawContours(dilate, [c], -1, 0, -1)
        # if (ar < 0.1):
        #     cv2.drawContours(dilate, [c], -1, 0, -1)
        # if (w==840):
        #     cv2.drawContours(dilate, [c], -1, 0, -1)
    # Dilate more
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    dilate = cv2.dilate(dilate, kernel, iterations=3)
    
    # Draw boxes
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h)
        #kontur = np.append(kontur,[[int(x),int(y)]])
        #kontur2 = np.append(kontur2,[[int(x),int(y),int(x)+int(w),int(y)+int(h)]])
        #if area > 10000:
            # for xx in range(0,int(int(area)/100)):
            #     kontur=np.append(kontur,[[int(x+(w/2)),int(y+(h/2))]])
            #     kontur2=np.append(kontur2,[[int(x),int(y),int(x)+int(w),int(y)+int(h)]])
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 5)
        # if (ar > 20):
        #     cv2.drawContours(dilate, [c], -1, 0, -1)
        # if (w > image.shape[1]*(0.9)):
        #     cv2.drawContours(dilate, [c], -1, 0, -1)
    
    
    
    
    
    font = cv2.FONT_HERSHEY_COMPLEX
    img2 = cv2.imread(rsm, cv2.IMREAD_COLOR)
      
    # Reading same image in another 
    # variable and converting to gray scale.
    img = cv2.imread(rsm, cv2.IMREAD_GRAYSCALE)
    
    img1 = img[:,0:bol1]
    imgg2 = img[:,bol1:]
    if bol==1:
        img=img1
    else:
        img=imgg2
    
    # Converting image to a binary image
    # ( black and white only image).
    _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    
    # Detecting contours in image.
    contours, _= cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours, _= cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Going through every contours found in the image.
    for cnt in contours :
        approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)
        # approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        # draws boundary of contours.
        #cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
        
        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel() 
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                if (bol!=1):
                    x = x + bol1+bolorta
                y = y + topmargin    
                if (bol==1):
                    kontur_1 = np.append(kontur_1,[[int(x),int(y)]])
                    kontur_1_2 = np.append(kontur_1_2,[[int(x),int(y),int(x),int(y)]])
                else:
                    kontur_2 = np.append(kontur_2,[[int(x),int(y)]])
                    kontur_2_2 = np.append(kontur_2_2,[[int(x),int(y),int(x),int(y)]])
                
    
      
                if(i == 0):
                    pass
                else:
                    pass# text on remaining co-ordinates.
                    #cv2.putText(img2, string, (x, y), font, 0.1, (0, 255, 0)) 
                    #cv2.circle(img2,(x,y), 5, (0,0,255), -1)
            i = i + 1
    
    if bol==1:
        kontur = kontur_1
        kontur2 = kontur_1_2
    else:
        kontur = kontur_2
        kontur2 = kontur_2_2       
        
    kontur_1 = kontur_1.reshape((-1,2))
    kontur_1_2 = kontur_1_2.reshape((-1,4))
    kontur_2 = kontur_2.reshape((-1,2))
    kontur_2_2 = kontur_2_2.reshape((-1,4))
    

    kontur = kontur.reshape((-1,2))
    kontur2 = kontur2.reshape((-1,4))
    
    kontur = np.float32(kontur)
    
    
    
    # import seaborn as sns
    # from matplotlib.colors import ListedColormap
    # from sklearn import neighbors, datasets
    # from sklearn.inspection import DecisionBoundaryDisplay
    
    # n_neighbors = 15
    
    # # import some data to play with
    # iris = datasets.load_iris()
    
    # # we only take the first two features. We could avoid this ugly
    # # slicing by using a two-dim dataset
    # X = kontur
    
    # # Create color maps
    # cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    # cmap_bold = ["darkorange", "c", "darkblue"]
    
    # for weights in ["uniform", "distance"]:
    #     # we create an instance of Neighbours Classifier and fit the data.
    #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    #     clf.fit(kontur[0].reshape((-1,1)),kontur[1].reshape((-1,1)))
    
    #     _, ax = plt.subplots()
    #     DecisionBoundaryDisplay.from_estimator(
    #         clf,
    #         X,
    #         cmap=cmap_light,
    #         ax=ax,
    #         response_method="predict",
    #         plot_method="pcolormesh",
    #         xlabel=kontur.feature_names[0],
    #         ylabel=kontur.feature_names[1],
    #         shading="auto",
    #     )
    
    #     # Plot also the training points
    #     sns.scatterplot(
    #         x=kontur[:, 0],
    #         y=kontur[:, 1],
    #         hue=iris.target_names[y],
    #         palette=cmap_bold,
    #         alpha=1.0,
    #         edgecolor="black",
    #     )
    #     plt.title(
    #         "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    #     )
    
    # plt.show()


    img_gray = gray
    template = cv2.imread('e.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    loc = np.where( res >= threshold)
    say= 0 
    oncekix=0
    oncekiy=0

    for pt in zip(*loc[::-1]):
        if  ( (abs((pt[0] + w)-oncekix)<4) and (abs((pt[1] + h)-oncekiy)<4) ):
            pass
        else:
            say=say+1
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            print(pt[0] + w,pt[1] + h)
        oncekix=(pt[0] + w)
        oncekiy=(pt[1] + h)
    print(say)    
    cv2.imwrite('res.png',image)


    
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(kontur,say,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    A = kontur[label.ravel()==0]
    B = kontur[label.ravel()==1]
    C = kontur[label.ravel()==2]
    # D = kontur[label.ravel()==3]
    # E = kontur[label.ravel()==4]
    # F = kontur[label.ravel()==5]
    print(label.ravel()==0)
     
    
    #Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(C[:,0],C[:,1])
    # plt.scatter(D[:,0],D[:,1])
    # plt.scatter(E[:,0],E[:,1])
    # plt.scatter(F[:,0],F[:,1])
    plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.show()

    if (bol==1):
        etiket_1 = np.hstack((kontur2,label))
        etiket_1 = pd.DataFrame(etiket_1,columns=['x','y','w','h','label'])
        etiket = etiket_1
        etiket0 = etiket.groupby('label').get_group(0)
        etiket1 = etiket.groupby('label').get_group(1)
        #etiket2 = etiket.groupby('label').get_group(2)
        group0 = [etiket0["x"].min(),etiket0["y"].min(), (etiket0["w"].max()) , (etiket0["h"].max()) ]
        group1 = [etiket1["x"].min(),etiket1["y"].min(), (etiket1["w"].max()) , (etiket1["h"].max()) ]
        #group2 = [etiket2["x"].min(),etiket2["y"].min(), (etiket2["w"].max()) , (etiket2["h"].max()) ]
        cv2.rectangle(imagereel, (int(group0[0]), int(group0[1])), (int(group0[2]), int(group0[3])), (36,0,12),10)
        cv2.rectangle(imagereel, (int(group1[0]), int(group1[1])), (int(group1[2]), int(group1[3])), (36,0,12),10)
        #cv2.rectangle(imagereel, (int(group2[0]), int(group2[1])), (int(group2[2]), int(group2[3])), (36,0,12),10)
    else:
        etiket_2 = np.hstack((kontur2,label))
        etiket_2 = pd.DataFrame(etiket_2,columns=['x','y','w','h','label'])
        etiket = etiket_2
        etiket3=etiket.groupby('label').get_group(0)
        etiket4=etiket.groupby('label').get_group(1)
        #etiket5=etiket.groupby('label').get_group(2)
        group3 = [etiket3["x"].min(), etiket3["y"].min(), (etiket3["w"].max()) , (etiket3["h"].max()) ]
        group4 = [etiket4["x"].min(), etiket4["y"].min(), (etiket4["w"].max()) , (etiket4["h"].max()) ]
        #group5 = [etiket5["x"].min(), etiket5["y"].min(), (etiket5["w"].max()) , (etiket5["h"].max()) ]
        cv2.rectangle(imagereel, (int(group3[0]), int(group3[1])), (int(group3[2]), int(group3[3])), (36,0,12),10)
        cv2.rectangle(imagereel, (int(group4[0]), int(group4[1])), (int(group4[2]), int(group4[3])), (36,0,12),10)
        #cv2.rectangle(imagereel, (int(group5[0]), int(group5[1])), (int(group5[2]), int(group5[3])), (36,0,12),10)
    
    
    
    
    
    cv2.imwrite(str(bol)+'1thresh.png', thresh)
    cv2.imwrite(str(bol)+'2dilate.png', dilate)
    cv2.imwrite(str(bol)+'3opening.png', opening)
    cv2.imwrite(str(bol)+'4image.png', image)
    cv2.imwrite(str(bol)+'6gray.png', gray)
    
    etiket0 = etiket0.reset_index()
    #for ii in range(etiket0['x'].count()):
    #    cv2.putText(dilate, str(etiket0.loc[ii,['w']].values),(int(etiket0.loc[ii,['x']].values), int(etiket0.loc[ii,['y']].values)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)

i=0    
for yaz in range(etiket_1['x'].count()):
    if (etiket_1.loc[i,'label']==0):
        renk=(0,0,255)
    elif (etiket_1.loc[i,'label']==1):
        renk=(0,255,0)
    elif (etiket_1.loc[i,'label']==2):
        renk==(255,0,0)
    cv2.circle(img2,(int(etiket_1.loc[i,'x']),int(etiket_1.loc[i,'y'])), 5, (0,0,255), -1)
    i=i+1

i=0
for yaz in range(etiket_2['x'].count()):
    if (etiket_2.loc[i,'label']==0):
        renk=(0,0,255)
    elif (etiket_2.loc[i,'label']==1):
        renk=(0,255,0)
    elif (etiket_2.loc[i,'label']==2):
        renk==(255,0,0)
    cv2.circle(img2,(int(etiket_2.loc[i,'x']),int(etiket_2.loc[i,'y'])), 5, (255,0,0), -1)
    i=i+1



cv2.imwrite('5imagereel.png', imagereel)
cv2.waitKey()
        


cv2.imwrite("a.jpg",img2)
cv2.imwrite("thres.jpg",threshold)
cv2.imwrite("img.jpg",img)





# data=pd.DataFrame({"x":kontur_1[:,0], "y":kontur_1[:,1]})

# from sklearn.cluster import KMeans
# wcss = [ ]
# for k in range(1,15 ) :
#     kmeans = KMeans(n_clusters = k)
#     kmeans.fit(data)
#     wcss.append(kmeans.inertia_)
# plt.figure()
# plt.plot(range(1,15), wcss)
# plt.xticks(range(1,15))
# plt.xlabel("Küme Sayısı ( K )")
# plt.ylabel("wcss")
# plt.show()


# k_ortalama = KMeans(n_clusters=6)
# kumeler = k_ortalama.fit_predict(data)
# data["label"] = kumeler
# plt.figure()
# plt.scatter(data.x[data.label == 0], data.y[data.label == 0], label ="küme1")
# plt.scatter(data.x[data.label == 1], data.y[data.label == 1], label ="küme2")
# plt.scatter(data.x[data.label == 2], data.y[data.label == 2], label ="küme3")
# plt.scatter(data.x[data.label == 3], data.y[data.label == 3], label ="küme4")
# plt.scatter(data.x[data.label == 4], data.y[data.label == 4], label ="küme5")
# plt.scatter(data.x[data.label == 5], data.y[data.label == 5], label ="küme6")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title ("6 - Ortalama Kümeleme Sonucu")


# from sklearn.cluster import AgglomerativeClustering
# hiyerarsi_kume = AgglomerativeClustering(n_clusters = 6, affinity = "euclidean")
# kume = hiyerarsi_kume.fit_predict(data)
# data["label"] = kume
# plt.figure()
# plt.scatter(data.x[data.label == 0 ] , data.y[data.label == 0] ,label ="küme1")
# plt.scatter(data.x[data.label == 1 ] , data.y[data.label == 1],label ="küme2")
# plt.scatter(data.x[data.label == 2 ] , data.y[data.label == 2] ,label ="küme3")
# plt.scatter(data.x[data.label == 3 ] , data.y[data.label == 3] ,label ="küme4")
# plt.scatter(data.x[data.label == 4 ] , data.y[data.label == 4] ,label ="küme5")
# plt.scatter(data.x[data.label == 5 ] , data.y[data.label == 5]  ,label ="küme6")
# plt.legend()
# plt.xlabel(" x ")
# plt.ylabel(" y ")
# plt.title (" Hiyerarşik Kümeleme Sonucu ")

# dbscan = DBSCAN(eps = 8, min_samples = 4).fit(kontur) # fitting the model
# labels = dbscan.labels_ # getting the labels
# plt.scatter(kontur[:, 0], kontur[:,1], c = labels, cmap= "plasma") # plotting the clusters
# plt.xlabel("Income") # X-axis label
# plt.ylabel("Spending Score") # Y-axis label
# plt.show() # showing the plot
# Z = np.float32(dilate.reshape((-1,3)))
# db = DBSCAN(eps=0.3, min_samples=100).fit(Z[:,:2])
# plt.imshow(np.uint8(db.labels_.reshape(img.shape[:2])))
# plt.show()




# for i in A:
#     cv2.putText(dilate, "A",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
# for i in B:
#     cv2.putText(dilate, "B",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
# for i in C:
#     cv2.putText(dilate, "C",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
# for i in D:
#     cv2.putText(dilate, "D",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
# for i in E:
#     cv2.putText(dilate, "E",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
# for i in F:
#     cv2.putText(dilate, "F",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
# for i in center:
#     cv2.putText(dilate, "M",(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)
#Now separate the data, Note the flatten()
