import streamlit
import pickle
import numpy

loaded_model = pickle.load(open('trained_lrmodel.sav','rb'))
#create function to handle predicition
def microwave_fault_prediction (user_input_data):
    #convert to array
    Input_array = numpy.asarray(user_input_data)
    Input_array_reshaped = Input_array.reshape(1, -1)
    make_prediction = loaded_model.predict(Input_array_reshaped)
    print(make_prediction)  # make_prediction= [10],  pos is 0

    if make_prediction== 0:
        return 'site  is up'
    elif make_prediction == 1:
        return'site  is down: fault: 1. inteference 2. misalignment 3. one of the odu is faulty'
    elif make_prediction == 2:
        return 'site  is down: fault: 1. NO power at remote site(A),2.ODU offline remote site(A)(check alarm \'IF cable open\') '
    elif make_prediction == 3:
        return 'site  is down: fault:1.ODu hunged at remote site(A), reset power at both sites(A,B)'
    elif make_prediction == 4:
        return 'site  is down: fault: 1. cascaded cable  faulty at hub  Site (B), 2.  ODU/IDU/If cable offline,at remote end'
    elif make_prediction == 5:
        return'site is down: fault: 1 ODU at hub site(B)degraded( reset ODU, reterminate IF cable,check alarm)'
    elif make_prediction == 6:
        return 'site  is down: faulty: if power is okay, odu burnt at either remote site (A) OR (B)'
    else:  # do feature elimination for data irrelevant to outcome
        return 'case 7: site status  cannot be determined by RSL data'

#construct interface for user data input
def main():
    #give a title
    streamlit.title('microwave fault detection web app')
    #get input data from user
    RSLA = streamlit.number_input('Site A Local end: enter RSL of the site, it must be negative number, input zero for no supervision',min_value=-99, max_value=0, value=-30, step=1,key= 'rsla')
    #key= 'rslb' is to distinguish two similar widgets 'text_input' in streamlit
    RSLB = streamlit.number_input('Site B Remote end: enter RSL of the Hub site, it  must be negative number, input zero for no supervision',min_value=-99, max_value=0, value=-30, step=1,key= 'rslb')
    #code for prediction
    detection =""   #declare this variable to hold result like empty list
    #mylist = []
    if streamlit.button('click here for  fault prediction'):
        detection=microwave_fault_prediction([RSLA,RSLB])
        #convert inputs into a single parameter using list [1,2]
        #microwave_fault_prediction ...call the function to process input
    streamlit.success(detection)
if __name__ == '__main__':
    main()


 # web app  on your desktop local host
#run  on your pycharm terminal ' streamlit run microwaveAPP.py '
# if using  command window ensure the path is correct..  c:\users\idom...\pycharm..\machin learing>
# that is pionting to your python file

