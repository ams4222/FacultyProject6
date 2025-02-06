# Faculty Project 6:
Strengthening OBFS4 & Tor Traffic Against Advanced Censorship

## Group Member Names:
Alyssa Sfravara, Connor Fox, Jacob Robertson, Trevor Jarrett

## Faculty Advisors:
Matt Wright and Nate Mathews

## Project Description: 
The project is motivated by our collaborators from the Naval Research Laboratory (NRL). The basic problem they want to investigate is the threat of censorship adversaries targeting the Tor anonymity network. Their concern is that censors may identify Tor users or Tor relays even when these connections are obfuscated with tools such as OBFS4 specifically by identifying patterns in the network traffic metadata that indicate that Tor communications are occurring inside the OBFS4 tunnel. The end goal of this broader project will be to develop a network traffic obfuscation defense that prevents this detection. 
With this in mind, we designed a component of this work as a capstone project in which the students are tasked to develop a dataset and a collection tool to produce that dataset that allows us to benchmark a classifier for detecting Tor tunneled under OBFS4. The dataset should contain "normal" OBFS4 tunneled traffic and OBFS4-tunneled Tor traffic, and our classifier's goal will be to determine if any given traffic sample contains Tor traffic in the tunnel. In particular, we are interested in replicating realistic traffic patterns inside the tunneled traffic. For this, we will want the data collection tool to be capable of replaying existing traffic samples (e.g., packet times and sizes) from real-world datasets such as the CAIDA or GTT23 network traffic dataset. After the students build a collection tool and collect a dataset, they will be tasked to analyze the dataset by building a basic classifier.


## About the Tool: 
