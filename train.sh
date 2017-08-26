#!/bin/bash
gcloud ml-engine local train --package-path=gan --module-name=gan.train -- \
--train_data_pattern=./train.tfrecords \
--generator_model=DCGANGenerator --discriminator_model=DCGANDiscriminator \
--train_dir=/tmp/kmlc_gan_train --num_epochs=500000 --start_new_model
