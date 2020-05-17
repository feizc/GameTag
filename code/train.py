import torch.utils.data.DataLoader as DataLoader
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import numpy as np
from data_preprocess import SpectrumRead
from dataloader import SpectrumLoader
from model import Discriminator, Generator, spectrum_refine
from evalue import tag_correct_discriminate, AverageMeter

# Data parameters
data_folder = '../data'  # the path of tag generation results
mgf_file_path = "../data/20100826_Velos2_AnMi_SA_HeLa_4Da_HCDFT.mgf"  # the path of spectrum mgf file
# the intersection labeled dataset can be obtained by data_label_union.py
# according to the identification results of PEAKS, MSFragger+ and Open-pFind.
spectrum_results_file_path = '../data/human.txt'
# train and evaluation datasets path, can be created automatically based on mgf file.
train_tag_feature_path = '../tag_feature_dataset.pkl'
train_tag_label_path = '../tag_label_dataset.pkl'
val_tag_feature_path = '../val_tag_feature_dataset.pkl'
val_tag_label_path = '../val_tag_label_dataset.pkl'

# Model parameters
check_point_D = 'discriminator_check_point.pth.tar'
check_point_flag = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
batch_size = 512

# Learning parameters
num_epoch = 50
lr = 1e-4
works = 4  # number of workers for accelerating the training.
print_freq = 200  # print loss every 200 batches.
best_accuracy = 0.
max_game_step = 5  # number of adversial game rounds.
mass_error = 0.1  # the mass difference for each amino acid inference.
# when find the optimal weights,
# we gradually reduce the search range for faster adjustation.
mass_error_reduction = 0.5
gap = 0.2


def main():

    global check_point_flag, check_point_D, num_epoch, best_accuracy, mass_error, gap

    # Read the mgf file to obtain the spectra.
    s = SpectrumRead(mgf_file_path)
    spectra_list = s.read_spectrum_data()
    print("Read the mgf file successfully!")

    # Fileter the unlabel spectra.
    spectra_list = spectrum_refine(spectra_list, spectrum_results_file_path)

    # generate the tag candidats with defeated parameters.
    print("The program is extracting the tag sequence conditioned on each spectrum: ")
    t = Generator(spectra_list, mass_error)
    t.process_spectrum_list()
    print("The extracted tag sequences are saved!")

    # save the tag candidates features.
    t.save_tag_feature(data_folder)

    # Label the generated tags.
    tag_correct_discriminate(data_folder, spectrum_results_file_path)

    # Initialize model or load checkpoint
    if check_point_flag is False:
        discriminator = Discriminator()
    else:
        check_point_D = torch.load(check_point_D)
        discriminator = check_point_D['model']

    # Move to GPU, if available.
    discriminator = discriminator.to(device)

    # model parameters optimization with classical SGD.
    optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)

    # Loss function.
    criterion = nn.CrossEntropyLoss().to(device)

    # training and evaluation dataloaders.
    train_loader = DataLoader(SpectrumLoader(train_tag_feature_path, train_tag_label_path),
                              batch_size=batch_size, shuffle=True, num_works=works)
    val_loader = DataLoader(SpectrumLoader(val_tag_feature_path, val_tag_label_path),
                            batch_size=batch_size, shuffle=True, num_works=works)

    print('The Tag Discriminator is ready to train.')
    for epoch in range(num_epoch):

        # One epoch's training
        discriminator_train(train_loader=train_loader,
                            model=discriminator,
                            criterion=criterion,
                            optimizer=optimizer,
                            epoch=epoch)

        # One epoch's validation
        recent_accuracy = discriminator_val(val_loader=val_loader,
                                            model=discriminator)

        # check if was best and save the best checkpoint.
        is_best = recent_accuracy > best_accuracy
        if is_best:
            filename = 'discriminator_check_point.pth.tar'
            state = {'model': discriminator,
                     'accuracy': best_accuracy}
            torch.save(state, filename)

    # After initialize the generator and discriminator, we further incorporate the adversial learning to
    # fine-tune the model for better performance.
    # Usually 2 iterative steps is enough for performance metric convergence.
    print('The Adversial Game is ready to play!')
    for i in range(max_game_step):

        # adjust the tag generator parameters conditioned on the trained discriminator.
        mass_error = generator_adjust(mass_error, discriminator, spectra_list, val_loader)

        # re-train the dicriminator and store the best model conditioned on current tag generator.
        for epoch in range(num_epoch):

            discriminator_train(train_loader=train_loader,
                                model=discriminator,
                                criterion=criterion,
                                optimizer=optimizer,
                                epoch=epoch)
            recent_accuracy = discriminator_val(val_loader=val_loader,
                                                model=discriminator)
            is_best = recent_accuracy > best_accuracy
            if is_best:
                filename = 'discriminator_check_point.pth.tar'
                state = {'model': discriminator,
                         'accuracy': best_accuracy}
                torch.save(state, filename)


def generator_adjust(mass_error, discriminator, spectra_list, val_loader):
    global gap, mass_error_reduction, best_accuracy

    best_mass_error = mass_error
    gap = mass_error_reduction * gap
    mass_gap = gap * mass_error

    # grid search under the dynamic parameter range.
    for error in np.arange(mass_error-mass_gap, mass_error+mass_gap, 0.5*mass_gap):

        # re-generate the candidate tag features.
        t = Generator(spectra_list, error)
        t.process_spectrum_list()
        t.save_tag_feature(data_folder)
        tag_correct_discriminate(data_folder, spectrum_results_file_path)

        # calculate the accuracy for the tag generator.
        current_accuracy = discriminator_val(val_loader, discriminator)

        # store the best model conditioned on current discriminator.
        if current_accuracy > best_accuracy:
            best_mass_error = error

        # The pre-defined stopping criterion.
        if abs(best_accuracy-current_accuracy) < 0.001:
            break

    return best_mass_error


# train the tag discriminator.
def discriminator_train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()

    for i, (tag_feature, tag_label) in enumerate(train_loader):

        tag_feature = tag_feature.to(device)
        tag_label = tag_label.to(device)

        # Forward prop.
        scores = model(tag_feature)

        # calculate the loss.
        loss = criterion(scores, tag_label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights.
        optimizer.step()
        losses.update(loss.item())

        # print the real-time loss.
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(train_loader), loss=losses))


# Test the performance of discriminator.
def discriminator_val(val_loader, model):
    model.eval()
    accuracy = AverageMeter()
    with torch.no_grad():
        for i, (tag_feature, tag_label) in enumerate(val_loader):
            tag_feature = tag_feature.to(device)
            tag_label = tag_label.to(device)
            scores = model(tag_feature)

            # compare the prediction label and true label to compute the accuracy.
            predict_idx = torch.max(scores, 1)[1]
            idx = torch.max(tag_label, 1)[1]
            assert idx.size(0) == predict_idx.size(0)
            accuracy_tmp = sum(predict_idx == idx).item() / idx.size(0)

            # We provide the real-time accuray and total accuracy to observe the training process.
            accuracy.update(accuracy_tmp)
    return accuracy.avg


if __name__ == '__main__':
    main()




