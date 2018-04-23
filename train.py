import time
import torch
from torch.autograd import Variable
import numpy as np
import os

torch.set_num_threads(16)


class Trainer:
    def __init__(self, cuda, model, train_loader, val_loader, loss, optimizer,
                 n_epochs, n_save, n_print, learning_rate, is_validation, lr_decay_every, lr_decay_ratio,
                 is_auto_adjust_rate, lr_adjust_every, out_dir):
        """
        Initialization for Trainer of FCN models for image segmentation.
        :param cuda: True is cuda available.
        :param model: The end-to-end FCN models.
        :param train_loader: The training dataset loader.
        :param val_loader: The validation dataset loader.
        :param loss: The loss function
        :param optimizer: The optimizer.
        :param n_epochs: The max number for epochs.
        :param n_save; Save every n_save iterations.
        :param n_print: Print every n_print iterations.
        :param learning_rate: The learning rate.
        :param is_validation: If using validation.
        :param lr_decay_every: learning rate decay every lr_decay_every steps.
        :param lr_decay_ratio: learning rate decay ratio.
        :param is_auto_adjust_rate: If automatically adjusting the learning rate.
        :param lr_adjust_every: The length to check and adjust the learning rate.
        :param out_dir: Output directory.


        """
        self.cuda = cuda
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = loss
        self.n_epochs = n_epochs
        self.n_save = n_save
        self.n_print = n_print
        self.optimizer = optimizer
        self.n_iter = 0
        self.is_validation = is_validation
        self.learning_rate = learning_rate
        self.lr_decay_every = lr_decay_every
        self.lr_decay_ratio = lr_decay_ratio
        self.is_auto_adjust_rate = is_auto_adjust_rate
        self.lr_adjust_every = lr_adjust_every
        self.out_dir = out_dir
        self.log_file = os.path.join(self.out_dir, 'log.txt')
        self.checkpoint_dir = os.path.join(self.out_dir, 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        n_iter = 0
        loss_train_print = 0
        train_count = 0
        loss_val_print = 0
        val_count = 0
        count_inner = 0
        loss_train_rec = np.array([]).reshape(1, 0)
        for epoch in range(self.n_epochs):
            tic = time.time()
            for img, seg, _ in self.train_loader:
                n_iter += 1
                self.optimizer.zero_grad()
                if self.cuda:
                    img, seg = img.cuda(), seg.cuda()
                img, seg = Variable(img), Variable(seg)
                output = self.model(img)

                # forward
                loss = self.loss(output, seg)
                loss_train_val = float(loss.data)
                loss_train_print += loss_train_val
                train_count += 1

                # backward
                loss.backward()

                # update parameter
                self.optimizer.step()

                # validation
                if self.is_validation:
                    loss_val = self.validation_loss()
                    loss_val_print += loss_val
                    val_count += 1

                count_inner += 1
                # restore the loss of most recent
                # loss_train_rec = np.append(loss_train_rec, [loss_train_print / train_count])

                # if n_iter > 2 * self.lr_adjust_every:
                #     loss_train_rec = loss_train_rec[1:]

                # learning rate decay/auto adjust
                # if self.is_auto_adjust_rate and n_iter >= 2 * self.lr_adjust_every and count_inner > 1000 and self.learning_rate > 1e-5:
                #     std_of_last = np.std(loss_train_rec[int(self.lr_adjust_every):], axis=0)
                #     mean_of_last = np.mean(loss_train_rec[0:int(self.lr_adjust_every)])
                #     mean_of_current = np.mean(loss_train_rec[self.lr_adjust_every:])
                #     if abs(mean_of_current - mean_of_last) <= std_of_last * 1:
                #         self.learning_rate *= self.lr_decay_ratio
                #         print("Learning rate is automatically adjusted")
                #     count_inner = 0
                #
                # elif ((n_iter + 1) % self.lr_decay_every) == 0 and self.is_auto_adjust_rate:
                #     self.learning_rate *= self.lr_decay_ratio

                # print to log
                if ((n_iter + 1) % self.n_print) == 0:
                    toc = time.time()
                    print(
                        "epoch {:5d}\t| step {:5d}\t| loss_train {:.5f}\t| loss_val {:.5f}\t| lr {:.5f}\t| time {:.5f} "
                        .format(epoch, n_iter, loss_train_print / train_count, loss_val_print / (val_count + 1e-16),
                                self.learning_rate, toc - tic))
                    with open(self.log_file, 'a') as log:
                        print(
                            "epoch {:5d}\t| step {:5d}\t| loss_train {:.5f}\t| loss_val {:.5f}\t| lr {:.5f}\t| time {:.5f} "
                            .format(epoch, n_iter, loss_train_print / train_count,
                                    loss_val_print / (val_count + 1e-16), self.learning_rate, toc - tic), file=log)
                    tic = time.time()
                    loss_train_print = 0
                    train_count = 0
                    loss_val_print = 0
                    val_count = 0

                # save models
                if ((n_iter + 1) % self.n_save) == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, '%08d_model.pth' % n_iter))
                    torch.save({
                        'optimizer': self.optimizer.state_dict(),
                        'iter': n_iter,
                        'epoch': epoch,
                    }, os.path.join(self.checkpoint_dir, '%08d_optimizer.pth' % n_iter))
                    # with open(self.out_dir + '/checkpoint/configuration.pkl', 'wb') as pickle_file:
                    # pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

    def validation_loss(self):
        """
        Calculate validation loss
        :return: the average loss over validation set.
        """
        loss_val = 0
        count = 0
        for img, seg, _ in self.val_loader:
            if self.cuda:
                img = img.cuda()
                seg = seg.cuda()
            img, seg = Variable(img), Variable(seg)
            output_val = self.model(img)
            loss_val += self.loss(output_val, seg).data
            count += 1
        loss_val /= count
        return loss_val


def main():
    pass


if __name__ == '__main__':
    main()
