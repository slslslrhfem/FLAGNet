import matplotlib.pyplot as plt
def plot_val_loss(hist):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

def plot_recall_f1score(hist):
    fig, recall_ax = plt.subplots()

    recall_ax.plot(hist.history['recall'], 'y', label='recall')
    recall_ax.plot(hist.history['val_recall'], 'r', label='valid recall')

    recall_ax.plot(hist.history['f1score'], 'b', label='f1score')
    recall_ax.plot(hist.history['val_f1score'], 'g', label='valid f1score')

    recall_ax.plot(hist.history['precision'], 'c', label='precision')
    recall_ax.plot(hist.history['val_precision'], 'k', label='valid precision')

    recall_ax.set_xlabel('epoch')
    recall_ax.set_ylabel('score')

    recall_ax.legend(loc='upper left')

    plt.show()