import vrep                  #V-rep's library
import sys
import time
import numpy as np
import pandas as pd

      
vrep.simxFinish(-1) #It is closing all open connections with VREP
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:  #It is checking if connection is successful
    print ('Connected to remote API server' )  
else:
    print ('Connection not successful')
    sys.exit('Could not connect')
    
errorCode,J1_handle=vrep.simxGetObjectHandle(clientID,"J1",vrep.simx_opmode_oneshot_wait)
errorCode,J2_handle=vrep.simxGetObjectHandle(clientID,"J2",vrep.simx_opmode_oneshot_wait)
errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'distance_sensor',vrep.simx_opmode_oneshot_wait)
time.sleep(1)




N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['down', 'up']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
FRESH_TIME = 0.01    # fresh time for one move

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
       # show table
    return table

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

def get_env_feedback(S, A ,distance ,position,d):
    # This is how agent will interact with the environment
    k=0
    for i in range(3):    
        errorCode,detectionstate, sensorreadingvalue,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor (clientID,sensor_handle1,vrep. simx_opmode_streaming)
        time.sleep(0.03)
    #print ("sensorreadingvalue ",sensorreadingvalue[2])
    if A == 'up':    # move right
        if S == N_STATES - 1:   # terminate
            S_ =  N_STATES - 1
        else:
            S_ = S + 1
            if d==1:
                errorCode=vrep.simxSetJointTargetPosition(clientID,J1_handle,(position+5)*3.14159/180, vrep.simx_opmode_oneshot)
                
            else:
                errorCode=vrep.simxSetJointTargetPosition(clientID,J2_handle,(position+5)*3.14159/180, vrep.simx_opmode_oneshot)
                
    else:   # move down
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
            if d==1:
               errorCode=vrep.simxSetJointTargetPosition(clientID,J1_handle,(position-5)*3.14159/180, vrep.simx_opmode_oneshot)
               
            else:
               errorCode=vrep.simxSetJointTargetPosition(clientID,J2_handle,(position-5)*3.14159/180, vrep.simx_opmode_oneshot)
               
    k=np.sqrt(pow(sensorreadingvalue[2],2)+pow(sensorreadingvalue[1],2)+pow(sensorreadingvalue[0],2))
    k=round(k,2)
    print("distanceK",k)
    if ((distance<k) &(distance !=0)):
        R=1       
    if((distance>k)&(distance !=0)): 
        R=-1
    if((distance==k) or (distance==0)):
        R=0
    distance=k
    #time.sleep(0.5)
    return S_, R ,distance,position

def rl():
    # main part of RL loop
    q_table1 = build_q_table(N_STATES, ACTIONS)
    q_table2 = build_q_table(N_STATES, ACTIONS)
    S1,S2 = 0,0
    position1=85
    position2=85
    errorCode=vrep.simxSetJointTargetPosition(clientID,J1_handle,position1*3.14159/180, vrep.simx_opmode_oneshot)
    errorCode=vrep.simxSetJointTargetPosition(clientID,J2_handle,position2*3.14159/180, vrep.simx_opmode_oneshot)
    distance=0
    tt=0
    while(1):
        A1 = choose_action(S1, q_table1)
        S_1, R ,distance ,position1= get_env_feedback(S1, A1 ,distance ,position1,1)  # take action & get next state and reward
        q_predict1 = q_table1.loc[S1, A1]
        if S_1 !=  (N_STATES - 1):
            q_target1 = R + GAMMA * q_table1.iloc[S_1, :].max()   # next state is not terminal
        else:
            q_target1 = R     # next state is terminal

        q_table1.loc[S1, A1] += ALPHA * (q_target1 - q_predict1)  # update
        S1 = S_1 # move to next state
        
        A2 = choose_action(S2, q_table1)
        S_2, R ,distance ,position2 = get_env_feedback(S2, A2 ,distance,position2,2)  # take action & get next state and reward
        q_predict2 = q_table2.loc[S2, A2]
        if S_2 !=  (N_STATES - 1):
            q_target2 = R + GAMMA * q_table2.iloc[S_2, :].max()   # next state is not terminal
        else:
            q_target2 = R     # next state is terminal

        q_table2.loc[S2, A2] += ALPHA * (q_target2 - q_predict2)  # update
        S2 = S_2 # move to next state
        if (tt%50==0):
            print("11111",q_table1)
            print("22222",q_table2)
        tt+=1;
if __name__ == "__main__":
    q_table = rl()
