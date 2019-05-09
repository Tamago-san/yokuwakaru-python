from sklearn import datasets, externals, linear_model ,model_selection
import time

print("GET MINIST ",end="", flush=True)
mnist=datasets.fetch_mldata("MNIST original", data_home=".")
data,label = mnist.data, mnist.target
print("END GET-MINIST")


TRAIN_SIZE =500
TEST_SIZE = 100

t=model_selection.train_test_split(
    data, label, train_size=TRAIN_SIZE, test_size=TEST_SIZE)
train_data,test_data,train_label,test_label =t
print("DATA       :",data.shape)
print("TRAIN DATA :",train_data.shape)
print("TEST DATA  :",test_data.shape)
print("TRAIN DATA label-shape :",train_label.shape)
print("TEST DATA  label-shape :",test_label.shape)

print("LET'S TRAIN: ",end="",flush=True)
old=time.time()
model = linear_model.LogisticRegression().fit(train_data,train_label)
print(time.time()-old, "s")

externals.joblib.dump(model,"1004-01.model")

print("TEST RESULT :")
predict=model.predict(test_data)
count= [[0 for i in range(10)] for j in range(10)]
for i in range(TEST_SIZE):
    count[int(predict[i])][int(test_label[i])] += 1
print("  ANSWER       ",end="")
for i in range(10):
    print(" [ {0} ]".format(i), end="")
print()
for i in range(10):
    print("PREDICT [{}]   ".format(i), end="")
    for j in range(10):
        print("{0:6d}".format(count[i][j]) ,end="")
    print()
    
print("ACURACY RATE:",model.score(test_data,test_label)*100,"%")