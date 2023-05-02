Files in the drive

X.pkl -> All actors. In form of dict {'PXXX': np.array() of shape (videos, 300 (frames), 150 (75joints x 2actors))}
test_X.pkl -> Same format as X.pkl, but has only actors in the testing split of the SGN training
x_classic_ml.pkl -> All actors. Same form as X.pkl, but ran through the motion retargeting algorithm
X_action.pkl -> All actors, but the key in dict is action instead of actor "AXXX"