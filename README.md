# microwave-fault-detection prediction 
This project uses machine learning model to predictive likely faults in a microwave link of a telecommunication system. The model is trained with microwave received signal level(RSL) data between two interconnected telecommunication sites.The readings of each pair of RSL at both sites can be use to  predictive likely causes of failure of the transmission path. This is a valuable tool for trainee microwave Engineers

A telecommunication link is usually established using microwave radio to send signal from your local end to the remote end. The RSL is a vital sign of how healthy the link is.

Received signal level (RSL) is measure how well a radio can 'hear'information being transmitted from the far end. The number is usually preceded by minus(-) sign. An example of RSL reading will be -33. The closer to zero, the stronger the signal however too strong RSL can damaged the receiver. Also, too weak RSL,the telecommunication radiocannot detect any signal transmitted from the far end.A good value  of RSL is not too strong or too weak, usually between -30 to -40.

To predictive fault in the microwave radio link using this model, readings of each pair of RSL at both sites is taken from a remote office. if a site RSL cannot be read from the remote office, it is assume zero for this project. The model then uses the pair of new RSL readings to predict the likely source of failure in the telecomunication system 
