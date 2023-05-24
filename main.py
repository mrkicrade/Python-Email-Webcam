import cv2
import time
from emailing import send_email
import glob
import os
from threading import Thread

video = cv2.VideoCapture(0)
# check, frame = video.read()
#
# print(check)
# print(frame)
time.sleep(1)

first_frame = None
status_list = []
count = 1

def clean_folder():
    print('clean_folder function started')
    images = glob.glob('images/*.png')
    for image in images:
        os.remove(image)
    print('clean_folder function ended')
#
while True:
    status = 0
    check, frame = video.read()
    # cv2.imshow('My video', frame)

    # Pretvaramo frame u gray frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blurujemo sliku
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # cv2.imshow('My video', gray_frame_gau)
#
    if first_frame is None:
        first_frame = gray_frame_gau

    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
    # cv2.imshow('My video', delta_frame)

    # Klasifikujemo piskele preko 60 kao bele piksele. Kada naidjemo na takve pikseledodeljujemo im vrednost 255
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('My video', thresh_frame)

    # Sada čistimo dodatno sliku
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cv2.imshow('My video', dil_frame)

    # tražimo konture objekata na slici
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # istražujemo konture objekata. Ako kontura ima vrednost manju od 5000 piksela to je nebitna kontura
    for contour in contours:
        if cv2.contourArea(contour) < 3000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if rectangle.any():
            # uvodimo varijablu status i ona ima vrednost 1 kada imamo pravougaonike. U suprotnom ima vrednost 0
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1
            all_images = glob.glob('images/*.png')
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        email_thread.start()
        clean_thread.start()
    print(status_list)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()

# clean_thread.start()