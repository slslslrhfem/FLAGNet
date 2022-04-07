class Hyperparams:
    Minimum_time = 16#This value means we use nth note for minimum unit. Use Integer value!(multiple of 4. If not, it can occur error while training GAN because of convolutioning)
#Highly recommend you to use this value 24 or 16.(16th note and 24th note as smallest unit.)
    # Note that if you Change The Minimum time value, then You should do preprocessing again.

    GAN_epochs = 100
    GAN_BATCH_SIZE = 128
    GAN_learning_rate = 0.0001
    with_chords=True # If this is True, then model generate chords at each bar with musical rule-based method
    pitch_shift =1 # 0 for use notes not in scale.
    bar_length=8
    
