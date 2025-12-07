# VoR - Daily Meditation Scraper & Distributor

An automated system that scrapes daily meditations from Voices of Recovery and distributes them via social media and SMS/MMS.

## Overview

This project automatically:
- Scrapes daily meditation content from SAA
- Generates an accompanying image
- Posts to BlueSky (@ocisaa.org)
- Posts to X/Twitter
- Sends SMS/MMS to subscribers
- Runs via GitHub Actions workflows

## Features

### Social Media Distribution
- **Automated Daily Posts**: Scheduled via GitHub Actions (1 AM Pacific)
- **Multi-Platform**: Posts to BlueSky and X simultaneously
- **Visual Content**: Generates custom images for each meditation
- **Zero Maintenance**: Runs automatically without manual intervention

### SMS/MMS Distribution
- **Daily Text Messages**: Sends meditations to subscribers (8 AM Pacific)
- **Google Sheets Integration**: Manage subscribers via spreadsheet
- **MMS Support**: Includes meditation images in messages
- **Subscriber Management**: Active/Inactive status tracking
- **Admin Notifications**: Automatic error alerts via SMS
- **Opt-out Compliance**: Respects STOP/UNSUBSCRIBE requests

## Architecture

### Workflows

**Daily Meditation Poster** (runs at 1 AM Pacific)
- Scrapes meditation from SAA website
- Generates meditation image
- Posts to BlueSky and X
- Commits text and image files to repository

**Daily SMS Sender** (runs at 8 AM Pacific)
- Reads subscribers from Google Sheet
- Sends SMS with meditation text
- Sends MMS with meditation image
- Notifies admins of any errors

### Components

- `daily_poster.py` - Main scraping and social media posting script
- `sms_sender.py` - SMS/MMS distribution script
- Google Sheets - Subscriber database
- Twilio - SMS/MMS delivery service

## Configuration

### Required GitHub Secrets

**Social Media:**
- `BSKY_USERNAME` - BlueSky username
- `BSKY_APP_PASSWORD` - BlueSky app password
- `X_API_KEY` - X/Twitter API key
- `X_API_SECRET` - X/Twitter API secret
- `X_ACCESS_TOKEN` - X/Twitter access token
- `X_ACCESS_TOKEN_SECRET` - X/Twitter access token secret

**SMS Distribution:**
- `TWILIO_ACCOUNT_SID` - Twilio account SID
- `TWILIO_AUTH_TOKEN` - Twilio auth token
- `TWILIO_PHONE_NUMBER` - Twilio phone number
- `GOOGLE_CREDENTIALS_JSON` - Google Sheets service account credentials

### Google Sheets Setup

Create a sheet with the following columns:
- **Phone Number**: E.164 format (+1XXXXXXXXXX)
- **Name**: Subscriber name (optional)
- **Status**: Active/Inactive
- **Date Added**: When subscribed
- **Source**: How they subscribed (Manual, Text, etc.)
- **Date Unsubscribed**: When they unsubscribed (if applicable)
- **Is Admin**: Yes/No - receives error notifications

## Usage

### Running Locally

**Test social media posting:**
```bash
python daily_poster.py --test
```

**Test SMS sending:**
```bash
python sms_sender.py --test
```

**Run for specific date:**
```bash
python daily_poster.py 12-07
python sms_sender.py 12-07
```

### Managing Subscribers

Add subscribers to the Google Sheet with:
- Phone number (will auto-add +1 if missing)
- Name
- Status = "Active"
- Fill in other fields as appropriate

To make someone an admin (receives error notifications):
- Set "Is Admin" column to "Yes"

### Manual Workflow Triggers

Both workflows can be manually triggered from the GitHub Actions tab if needed.

## Future Plans

- Automated subscribe/unsubscribe via SMS webhooks
- Automatic sync with Twilio opt-out list
- Additional distribution channels
- Analytics and delivery tracking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## About

Created by JacksonS (steptwelve@icloud.com) to provide daily meditations to the recovery community through automated distribution.

Special thanks to Claude for assistance with development and implementation.

## Support

If you find this project helpful, consider supporting its operation:
- Venmo: [your handle]
- PayPal: [your handle]
- GitHub Sponsors: [pending approval]

SMS distribution costs approximately $3-5 per subscriber per year. Donations help keep this service free for the recovery community.
