import mediapipe as mp
import cv2
import numpy as np
from cnn import Model, DataGatherer
from Auto_Correct_SpellChecker import Auto_Correct
from GUI import GUI
from tkinter import *
from PIL import ImageTk, Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = None

#pre-trained saved model with 99% accuracy
classifier = Model.load_classifier('grayscale_classifier.h5')


def draw_region(image, center):
    cropped_image = cv2.rectangle(image, (center[0] - 130, center[1] - 130),
        (center[0] + 130, center[1] + 130), (0, 0, 255), 2)
    return cropped_image[center[1]-130:center[1]+130, center[0]-130:center[0]+130], cropped_image

def start_gui(title, size):
    gui = GUI(title, size)

    gui_frame = gui.create_frame(600, 600, 'ne', 1, 0, 'green')
    vid_label = Label(gui_frame)
    vid_label.grid()
    
    return gui, vid_label

def exit_app(gui, cap):
    gui.root.destroy()
    cap.release()


def update_frame(image, vid_label):
    image_fromarray = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image_fromarray)
    
    vid_label.imgtk = imgtk
    vid_label.config(image=imgtk)

def get_threshold(label_entrybox):
    value = label_entrybox.get('1.0', END)
    try:
        return float(value)
    except:
        return 0.95


def get_char(gesture):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    return Model.predict(classes, classifier, gesture)


def AddCharToWord(word, curr_char):
    temp_word = word
    if curr_char == 'space':
        #print(Auto_Correct(temp_word))
        temp_word = ""
    elif curr_char == 'del':
        temp_word = temp_word[0:-1]
        print('character has been deleted')
    elif curr_char != 'nothing':
        temp_word += curr_char.lower()
        print('character has been added: ', curr_char.lower())

    return [temp_word, curr_char]


def frame_video_stream(names, curr_char, prev_char, word, sentence, *args):
    kwargs = dict(zip(names, args))
    
    threshold = get_threshold(kwargs['th_box'])
    curr_char = curr_char
    prev_char = prev_char
    
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    update_frame(image, kwargs['vid_label'])

    image.flags.writeable = False
    results = kwargs['hands'].process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]

            
            center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped_img, full_img = draw_region(image, center)

            update_frame(full_img, kwargs['vid_label'])

            try:
                #print('from try')
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                gray = DataGatherer().edge_detection(gray)

                curr_char, pred = get_char(gray)
                char = cv2.putText(full_img, curr_char, (center[0]-135, center[1]-135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                char_prob = cv2.putText(full_img, '{0:.2f}'.format(np.max(pred)), (center[0]+60, center[1]-135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                update_frame(full_img, kwargs['vid_label'])

                kwargs['cc_box'].delete('1.0', 'end')
                kwargs['cc_box'].insert('end', curr_char)
                #print(curr_char)
                #compare the current char with the previous one and if matched, then won't add the current char
                #because the model catches the chars realy quick and if the below if statement removed,
                #the current char will be added endlessly to the word

                #also we use the threshold to prevent the meaningless characters to be added to the word
                #as the program catches the motion of the user's hand when the user changes the gesture(the motion between the gestures)
                #and the program thinks
                #it's a gesture and tries to match it with some letter but with low probability
                if (curr_char != prev_char) and (np.max(pred) > threshold):
                    #the below print statement is related to the formatter
                    #print(pred)
                    temp = AddCharToWord(word, curr_char)
                    kwargs['ow_box'].insert('end', curr_char)
                    
                    if (temp[0] == "") and (temp[1] != "del"):
                        sentence += Auto_Correct(word) + " "
                        kwargs['sent_box'].insert('end', Auto_Correct(word) + " ")
                        kwargs['ow_box'].delete('1.0', 'end')
                        kwargs['cw_box'].delete('1.0', 'end')
                        kwargs['cw_box'].insert('end', Auto_Correct(word))
                    word = temp[0]

                    prev_char = curr_char
            except:
                pass
    
    kwargs['vid_label'].after(1, frame_video_stream, names, curr_char, prev_char, word, sentence, *args)
    
    
    
def pipe_cam(gui, vid_label):
    
    curr_char = None
    prev_char = None
    word = ""
    sentence = ""

    
    
    
    #the predicted character won't be added to the word unless it's
    #probability is higher than the threshold
    #in places with good brightness and good camera the threshold can be a high value
    #otherwise it should be a low value and the reason for that is in places that meet
    #the above requirements, the model predict the letters with high probability to be
    #the correct letter the user ment to add
    threshold = float(0.95)

    

    #this formatter is to print the probability of the letters in readable
    #format just for the programmers if they want to see what are the probabilities looking like
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    
    global cap
    cap = cv2.VideoCapture(0)

    labels_num = 5
    labels = ['threshold', 'current char', 'original word', 'corrected word', 'sentence']

    Labels, entryboxes = gui.create_labels(labels_num, labels, 'nw', 0, 0, y_spacing=0.06, create_entrybox_per_label=1)

    entryboxes['original word_entrybox'].config(width=18)
    entryboxes['corrected word_entrybox'].config(width=18)
    entryboxes['sentence_entrybox'].config(width=18, height=8)
    
    
    entryboxes['threshold_entrybox'].insert('end', threshold)
    th_entrybox = entryboxes['threshold_entrybox']


    cc_entrybox = entryboxes['current char_entrybox']


    ow_entrybox = entryboxes['original word_entrybox']


    cw_entrybox = entryboxes['corrected word_entrybox']


    sent_entrybox = entryboxes['sentence_entrybox']

    
    Exit_program_btn = gui.create_buttons(1, ['Exit'], 'center', 0.5, 0.9, command=lambda: exit_app(gui, cap))

    names = ['vid_label', 'hands', 'th_box', 'cc_box', 'ow_box', 'cw_box', 'sent_box']
    with mp_hands.Hands(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5,
            max_num_hands=1) as hands:
        
            frame_video_stream(names, curr_char, prev_char, word, sentence, vid_label,
                               hands,  th_entrybox, cc_entrybox, ow_entrybox, cw_entrybox, sent_entrybox)
            gui.root.mainloop()


title = "Sign Language Recognition GUI"
size = "1100x1100"

gui, vid_label = start_gui(title, size)

pipe_cam(gui, vid_label)



