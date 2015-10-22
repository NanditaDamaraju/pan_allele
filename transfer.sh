scp -i ~/amazon_keys/Pan-allele-class-I.pem ~/Intern/mhclearn/py/pan_allele/files/*.* ubuntu@ec2-52-10-60-215.us-west-2.compute.amazonaws.com:~/py/pan_allele/files/
scp -i ~/amazon_keys/Pan-allele-class-I.pem ~/Intern/mhclearn/py/pan_allele/*.py ubuntu@ec2-52-10-60-215.us-west-2.compute.amazonaws.com:~/py/pan_allele/
scp -i ~/amazon_keys/Pan-allele-class-I.pem ~/Intern/mhclearn/py/Spearmint/*.* ubuntu@ec2-52-10-60-215.us-west-2.compute.amazonaws.com:~/hyperparam
scp -i ~/amazon_keys/Pan-allele-class-I.pem -r ~/Intern/mhclearn/py/pan_allele/paper_data/ ubuntu@ec2-52-10-60-215.us-west-2.compute.amazonaws.com:~/py/pan_allele/paper_data
