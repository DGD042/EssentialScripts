# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 04/02/2019
#______________________________________________________________________________
#______________________________________________________________________________
'''

The functions given on this package allow the user to manipulate and create
functions from the computer.

Usage::

    >>> from Utilities import Utilities as utl
    >>> # Create a folder
    >>> utl.CrFolder(<Path_inside_computer>)

'''
# ------------------------
# Importing Modules
# ------------------------ 
# System
import sys
import os
import glob as gl
import re
import operator as op
import warnings
import subprocess
import platform
import time

# Email
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

# System
def ShowError(fn,cl,msg):
    '''
    DESCRIPTION:

        This function manages errors, and shows them. 
    _______________________________________________________________________
    INPUT:
        :param fn:  A str, Function that produced the error.
        :param cl:  A str, Class that produced the error.
        :param msg: A str, Message of the error.
    _______________________________________________________________________
    OUTPUT:
    '''

    raise Exception('ERROR: Function <'+fn+'> Class <'+cl+'>: '+msg)

def CrFolder(path):
    '''
    DESCRIPTION:
    
        This function creates a folder in the given path, if the path does 
        not exist then it creates the path itself
    _______________________________________________________________________

    INPUT:
        :param path: A str, Path that needs to be created.
    _______________________________________________________________________
    OUTPUT:
        :return: This function create all the given path.
    '''
    if path != '':
        # Verify if the path already exists
        if not os.path.exists(path):
            os.makedirs(path)

    return

def GetFolders(path):
    '''
    DESCRIPTION:
    
        This function gets the folders and documents inside a 
        specific folder.
    _______________________________________________________________________

    INPUT:
        :param path: A str, Path where the data would be taken.
    _______________________________________________________________________
    OUTPUT:
        :return R: A List, List with the folders and files inside 
                           the path.
    '''
    R = next(os.walk(path))[1]
    return R

def start_mail(username,password,mailServer='outlook'):
    '''
    DESCRIPTION:
        This code sends an email.
    ____________________________________________________________
    INPUT:
        :param username: str, username of the email.
        :param password: str, password of the mail.
        :param mailServer: str, mail server
                
            Implemented: Outlook,
    ____________________________________________________________
    OUTPUT:
        Send an email.
    '''
    # Starting mailing
    if mailServer.lower() == 'outlook':
        mailServer = smtplib.SMTP('smtp-mail.outlook.com', 587)
    if mailServer.lower() == 'gmail':
        mailServer = smtplib.SMTP('smtp.gmail.com', 587)
    if mailServer.lower() == 'vanderbilt':
        mailServer = smtplib.SMTP('smtpauth.vanderbilt.edu', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.login(username,password)
    return mailServer

def send_mail(send_to,subject,text,mailServer,send_from,files=None):
    '''
    DESCRIPTION:
        This code sends an email.
    ____________________________________________________________
    INPUT:
        :param send_to: str, receiver direction.
        :param subject: str, subject of the email.
        :param text: str, Text of the email.
        :param files: str, path/to/the/file.ext

    ____________________________________________________________
    OUTPUT:
        Send an email.
    '''

    if isinstance(send_to,str):
        send_to = [send_to]
    elif isinstance(send_to,list):
        send_to = send_to
    else:
        ValueError('send_to must be a str or a list')

    msg = MIMEMultipart()
    msg['From'] = send_from 
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))
    mailServer.sendmail(send_from,send_to,msg.as_string())

    
    return

def toc(time1):
    dif = time.time() - time1
    if dif >= 3600*24:
        print(f'====\t{dif/3600/24:.4f} days\t ====')
    elif dif >= 3600:
        print(f'====\t{dif/3600:.4f} hours\t ====')
    elif dif >= 60:
        print(f'====\t{dif/60:.4f} minutes\t ====')
    else:
        print(f'====\t{dif:.4f} seconds\t ====')
    return
