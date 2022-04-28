import pygame as py
import _thread
import time
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import multiprocessing
import  os
import time
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure




window_width=960
window_height=480
image_width=int(window_width*0.5)
image_height=int(window_height*0.5)


video_name='2021040004'
video_name='test'
# video_name='2021040019'

img_list=sorted(os.listdir(os.path.join(video_name,'save_img_new')))

txt_path=os.path.join(video_name,video_name+'_pred.txt')

# print(txt_path)
f = open(txt_path,'r')
t = f.read()
f.close()
import re  # 导入re库
pattern = re.compile(r'frame_path:(.*)')
result = pattern.findall(t)
# frame_list = list(map(int, result))
frame_list=result

pattern = re.compile(r'TOP4_texts:(.*)')
result = pattern.findall(t)
wrong_list=[]
for w in result:
    w=w.split(',')
    for k in w:
        wrong_list.append(k.split('\'')[-2].replace(' ',''))

pattern = re.compile(r'TOP4_scores:(.*)')
result = pattern.findall(t)
score_list=[]
for w in result:
    w=w.split(',')
    for k in w:
        score_list.append(round(float(k.replace('[','').replace(']','').replace(' ','')),2))


pattern = re.compile(r'TOP4_degree(.*)')
result = pattern.findall(t)
degree_list=[]
for w in result:
    w=w.split(',')
    for k in w:
        degree_list.append(k.replace('[','').replace(']','').replace(' ',''))



pattern = re.compile(r'clock_pos(.*)')
result = pattern.findall(t)
clock_list=[]
for w in result:
    w=w.split(',')
    for k in w:
        clock_list.append(k.split('\'')[-2].replace(' ',''))




pattern = re.compile(r'frame_index:(.*)')
result = pattern.findall(t)
# frame_list = list(map(int, result))
index_list=[int(x) for x in result]




pattern = re.compile(r'video_second:(.*)')
duration = int(float(pattern.findall(t)[0]))



txt_path=os.path.join(video_name,'nms_'+video_name+'_pred.txt')

f = open(txt_path,'r')
t = f.read()
f.close()
pattern = re.compile(r'frame_index:(.*)')
result = pattern.findall(t)
# frame_list = list(map(int, result))
nms_index_list=[int(x) for x in result]



pattern = re.compile(r'video_name:(.*)')
vide_name = pattern.findall(t)[0].split('/')[-1]


pattern = re.compile(r'total run time:(.*)')
run_time = str(int(float(pattern.findall(t)[0])))


text_end='Video Name: '+vide_name+'\n'+'Video Time: '+str(duration)+' seconds\n'+'Run Time: '+run_time+' seconds\n'+'Defect Num: '+str(len(nms_index_list))+'\n'

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(111)
width=4
rect1 = plt.Rectangle((0.0, width), duration,width, edgecolor='black',facecolor='none',linewidth=5, alpha=0.6)
rect2 = plt.Rectangle((0.0, width+width), duration,width, edgecolor='black',facecolor='none',linewidth=5, alpha=0.6)

ax.add_patch(rect1)
ax.add_patch(rect2)
plt.text(0.0, width-2, '0', fontsize=15)
plt.text(duration, width-2, str(duration), fontsize=15)
plt.text(0.0, width+width+width+2, 'Result', fontsize=15)

plt.text(0.0-15, width+width+width*0.5, 'Raw:', fontsize=15,color='r')
plt.text(0.0-15, width+width*0.5, 'NMS:', fontsize=15,color='g')

for sg in index_list:
    rect = plt.Rectangle((0.0+ sg , width*2), 1, width, color='r', alpha=0.3)
    ax.add_patch(rect)


for sg in nms_index_list:
    rect = plt.Rectangle((0.0+ sg , width), 1, width, color='g', alpha=0.3)
    ax.add_patch(rect)

plt.xlim(0,duration)
plt.ylim(3,14)
plt.axis('off')
plt.savefig('show.png')
# plt.show()

global start
start=False

global time_score_list
time_score_list=[]


global flag_idx

flag_idx=0

def Button_text_switch1():
    global start
    start=True
    global flag_idx
    flag_idx=bar.get()
    global time_score_list
    time_score_list=[]
def Button_text_switch2():
    global start
    start=False



#图像转换，用于在画布中显示
def tkImage(idx):
    flag=0
    
    frame=cv2.imread(os.path.join(video_name,'save_img_new',img_list[idx]))
    frame=cv2.resize(frame, (image_width, image_height))

    # print(os.path.join('test','save_img_new',img_list[idx]))
    cvimage1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if os.path.exists(os.path.join(video_name,img_list[idx])):
        frame=cv2.imread(os.path.join(video_name,img_list[idx]))
        frame=cv2.resize(frame, (image_width, image_height))
        frame=cv2.putText(frame, 'Defect', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cvimage2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flag=1
        text_=wrong_list[frame_list.index(img_list[idx])*4 :  frame_list.index(img_list[idx])*4+4]
        text_2=score_list[frame_list.index(img_list[idx])*4 :  frame_list.index(img_list[idx])*4+4]
        text_3=degree_list[frame_list.index(img_list[idx])*4 :  frame_list.index(img_list[idx])*4+4]
        text_4=clock_list[frame_list.index(img_list[idx])*4 :  frame_list.index(img_list[idx])*4+4]
        text='State: Defect\nTime:{:5d} seconds\nClass: '.format(idx)
        # for k in text_:
        #     text=text+"{: >7s}".format(k)+' '
        
        # text=text+'\nSorce: '
        sum_score=0
        for k in text_2:
            text=text+"{: >8s}".format(str(k))+' '
            sum_score=sum_score+float((k))

        # text=text+'\nDegree:'
        # for k in text_3:
        #     text=text+"{: >7s}".format(str(k))+' '

        # text=text+'\nClock: '
        # for k in text_4:
        #     text=text+"{: >7s}".format(str(k).lower())+' '
        

        # text=text+''
        # global time_score_list
        time_score_list.append(sum_score)

        contacts.append((str(idx),text_,text_2,text_3,text_4))
        


        
    else:
        frame=cv2.imread(os.path.join(video_name,'save_img_new',img_list[idx]))
        
        frame=cv2.resize(frame, (image_width, image_height))
        frame=cv2.putText(frame, 'Normal', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cvimage2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        text='State: Normal\nTime:{:5d} seconds'.format(idx)

        # global time_score_list
        time_score_list.append(random.random()*0.15)
    

    cvimage=cv2.hconcat([cvimage1,cvimage2])
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width*2, image_height),Image.ANTIALIAS)
    tkImage1 =  ImageTk.PhotoImage(image=pilImage)
        
    return tkImage1,flag,text


#图像的显示与更新
def video():
    
    global flag_idx
    picture1,flag,text=tkImage(flag_idx)
    canvas1_2.create_image(0,0,anchor='nw',image=picture1)  
    # canvas_table.create_text(10,10,anchor='nw',text=text,font=('Times',15))

    n = image_width / duration * flag_idx


    while True:
        
        if flag_idx >=len(img_list):
            frame=cv2.imread('show.png')
            frame=cv2.resize(frame, (image_width*2, image_height))
            cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(cvimage)
            pilImage = pilImage.resize((image_width*2, image_height),Image.ANTIALIAS)
            tkImage1 =  ImageTk.PhotoImage(image=pilImage)
            canvas1_2.create_image(0,0,anchor='nw',image=tkImage1)   

            canvas_control.delete('t') 
            text_end='Time: '+str(flag_idx)+' seconds'
            canvas_control.create_text(image_width/2,5,anchor='nw',text=text_end,font=('Times',15),tags='t')

            for k in nms_index_list:
                                
                tv.tag_configure(str(k-1), background='white', foreground="red")
            
            # fill_line = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
            # fill_line1 = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")
            # n = image_width / duration * flag_idx
            # canvas_control.coords(fill_line1, (0, 100, n, 105))
            # canvas_control.coords(fill_line, (0, 100, n, 105))


            win.update_idletasks()  #最重要的更新是靠这两句来实现
            win.update()
        else:
            if start==False:
                # print(start)
                # inser_data()
                canvas_control.delete('t') 
                text_end='Time: '+str(flag_idx)+' seconds'
                canvas_control.create_text(image_width/2,5,anchor='nw',text=text_end,font=('Times',15),tags='t')


                # fill_line = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
                # fill_line1 = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")
                # n = image_width / duration * flag_idx
                # canvas_control.coords(fill_line1, (0, 100, n, 105))
                # canvas_control.coords(fill_line, (0, 100, n, 105))

                win.update_idletasks()  #最重要的更新是靠这两句来实现
                win.update()
            else:
                
                # canvas_table.delete('all') 

                picture1,flag,text=tkImage(flag_idx)
                canvas1_2.create_image(0,0,anchor='nw',image=picture1)  
                

                f_plot.clear()#刷新
                plt.xlabel('Time',size = 20)
                plt.ylabel('Defect Score',size =20)
                plt.ylim((0,1))
                # print(len(time_score_list),time_score_list)
                plt.plot(range(len(time_score_list)),time_score_list)
                # plt.grid(True)#网格
                canvas_plot.draw()
                

                # canvas_table.create_text(10,10,anchor='nw',text=text,font=('Times',15))

                # fill_line = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
                n = image_width / duration * flag_idx
                canvas_control.coords(fill_line, (0, 100, n, 105))
                
                canvas_control.delete('t') 
                text_end='Time: '+str(flag_idx)+' seconds'
                canvas_control.create_text(image_width/2,5,anchor='nw',text=text_end,font=('Times',15),tags='t')

                flag_idx =flag_idx+1
                    # break
                if flag==1:
                    time.sleep(0.5)
                    inser_data()
                else:
                    time.sleep(0.01)
                win.update_idletasks()  #最重要的更新是靠这两句来实现
                win.update()
           
    win.mainloop()

'''布局'''
win = tk.Tk()
win.title('VideoPipe')
win.geometry(str(window_width)+'x'+str(window_height))



#图像及画布
fig = plt.figure(figsize=(2.4,0.65),dpi=100)#图像比例
f_plot =fig.add_subplot(111)#划分区域
canvas_plot = FigureCanvasTkAgg(fig,win)
canvas_plot.get_tk_widget().place(x=0,y=image_height)#放置位置
canvas_plot




    
canvas_table =Canvas(win,bg='white',width=image_width+25,height=image_height)
canvas_table.place(x=image_width-25,y=image_height)  


canvas_control =Canvas(win,bg='white',width=image_width-30,height=image_height-130)
canvas_control.place(x=0,y=image_height+130)  
fill_line = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
fill_line1 = canvas_control.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")

bn1=Button(win, text='Start', command=Button_text_switch1)
bn2=Button(win, text='Pause', command=Button_text_switch2)

button1_window = canvas_control.create_window(0, 0, anchor=NW, window=bn1)
button2_window = canvas_control.create_window(image_width/3, 0, anchor=NW, window=bn2)


canvas1_2 =Canvas(win,bg='white',width=image_width*2,height=image_height)
canvas1_2.place(x=0,y=0)

tk.Label(win, text='Time Line:', bg='white').place(x=0, y=image_height*1.86)
tk.Label(win, text=str(duration),bg='white').place(x=image_width-70, y=image_height*1.86)
 

# 实例化控件，设置表头样式和标题文本
columns = ("name", "tel", "email", "company", "CLock")
headers = ("Time", "Class", "Score", "Degree", "CLock")
widthes = (60, 90, 125, 65,160)
from tkinter import ttk
tv = ttk.Treeview(canvas_table, show="headings", columns=columns)

def fixed_map(option):
    # Returns the style map for 'option' with any styles starting with
    # ("!disabled", "!selected", ...) filtered out
    # style.map() returns an empty list for missing options, so this should
    # be future-safe
    return [elm for elm in style.map("Treeview", query_opt=option)
            if elm[:2] != ("!disabled", "!selected")]
            
style = ttk.Style()
style.map("Treeview",
                  foreground=fixed_map("foreground"),
                  background=fixed_map("background"))


for (column, header, width) in zip(columns, headers, widthes):
    tv.column(column, width=width, anchor="nw")
    tv.heading(column, text=header, anchor="nw")


contacts=[]
def inser_data():
    # tv.delete()
    # print(contacts)
    """插入数据"""
    for i, person in enumerate(contacts):
        if i==len(contacts)-1:
            tv.insert('', i, values=person,tag=str(person[0]))
            # print(person[0],nms_index_list)
                # print(person[0])

tv.pack()


bar=tk.Scale(canvas_control, from_=0, to=duration,bg='white', length=image_width-35, orient="horizontal")
bar.place(x=0,y=30) 




if __name__ == '__main__': 
    p1 = multiprocessing.Process(target=video)
    p1.start()