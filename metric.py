import cv2
import numpy as np
import utils
imgs = list(utils.load_imgs('img'))
'''
ans = (imgs[0] >= 0.5).astype(np.uint8) * 255
pred = (imgs[1] >= 0.5).astype(np.uint8) * 255
for img in imgs:
    img = (img >= 0.5).astype(np.uint8) * 255
    print(np.mean(img))

    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, connectivity)

    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    print('--------')
    print(num_labels)
    cv2.imshow('i',img); cv2.waitKey(0)
    for i in range(num_labels):
        print('label is', i)
        cv2.imshow('i',(labels==i).astype(np.uint8)*255); cv2.waitKey(0)
    print(np.unique(labels))
    print(labels.dtype)
'''
def intersection_table(ans, num_ans_labels, pred, num_pred_labels):
    '''
    row: ans label
    col: pred label
    content: number of pixels in intersection between ans and pred  

    NOTE 
    Table always has label 0. But it's meaningless.
    Just for convenient indexing.
    '''
    assert ans.shape == pred.shape
    itab = np.zeros((num_ans_labels,num_pred_labels),dtype=int)
    for ans_label in range(1,num_ans_labels):
        for pred_label in range(1,num_ans_labels):
            ans_component = (ans == ans_label)
            pred_component = (pred == pred_label)
            intersection = ans_component * pred_component
            num_intersected = np.sum(intersection.astype(int))
            itab[ans_label,pred_label] = num_intersected
    return itab

'''
cv2.imshow('a',ans_component.astype(np.float32))
cv2.imshow('p',pred_component.astype(np.float32))
cv2.imshow('i',intersection.astype(np.float32)); cv2.waitKey(0)
print(ans_component.astype(int))
print(pred_component.astype(int))
print(intersection.astype(int))
'''
    
import unittest
class test_itable(unittest.TestCase):
    def test_unmatched_shape(self):
        ans = np.ones((3,2))
        pred = np.ones((2,2))
        self.assertRaises(AssertionError, 
                          intersection_table, ans, 0, pred, 0)

    def test_simple_case(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,2,2,0,0,0,0,0,0,0],
            [0,0,2,2,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        pred = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,0,2,2,0,3,3,0],
            [0,0,1,1,0,2,2,0,3,3,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        expected_itab = np.array([
            [0,0,0,0],
            [0,4,0,0],
            [0,0,0,0],
        ],dtype=int)

        #cv2.imshow('a',ans.astype(np.float32))
        #cv2.imshow('p',pred.astype(np.float32)); cv2.waitKey(0)
        actual_itab = intersection_table(ans,len(np.unique(ans)), 
                                         pred,len(np.unique(pred)))
        self.assertEqual(actual_itab.tolist(),
                         expected_itab.tolist())

    def test_many_value_in_1row_case(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,3,3,0,0,0,0,0,0,0,0],
            [0,3,3,0,4,4,0,0,0,0,0],
            [0,0,0,0,0,4,0,0,0,5,5],
            [0,0,0,0,0,0,0,0,0,5,5],
        ],dtype=np.uint8)
        pred = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,0,0,0],
            [0,0,0,0,0,1,1,0,2,2,0],
            [0,3,3,3,0,0,0,0,2,0,0],
            [0,3,3,3,0,0,0,0,0,0,0],
            [0,3,3,3,0,0,4,4,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        expected_itab = np.array([
            [0,0,0,0,0,0],
            [0,4,3,0,2,0],
            [0,0,0,9,0,0],
            [0,0,0,0,0,4],
            [0,0,0,0,0,3],
            [0,0,0,0,0,0],
        ],dtype=int)

        actual_itab = intersection_table(ans,len(np.unique(ans)), 
                                         pred,len(np.unique(pred)))


if __name__ == '__main__':
    unittest.main()
