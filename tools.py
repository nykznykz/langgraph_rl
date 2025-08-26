"""Tools for the email agent example."""

def search_inbox(query: str) -> str:
    """Search emails matching query.
    
    Args:
        query: The search query to find matching emails
        
    Returns:
        A string describing the found emails
    """
    # Mock implementation - in a real scenario this would connect to an email API
    mock_emails = {
        "john": ["Email 1: Quarterly Report from John", "Email 2: Follow-up on Q4 numbers from John"],
        "budget": ["Email 3: Budget discussions for next quarter", "Email 4: Budget approval needed"],
        "sarah": ["Email 5: Latest updates from Sarah", "Email 6: Sarah's weekly summary"]
    }
    
    query_lower = query.lower()
    found_emails = []
    
    for keyword, emails in mock_emails.items():
        if keyword in query_lower:
            found_emails.extend(emails)
    
    if found_emails:
        return f"Found emails matching '{query}':\n" + "\n".join(f"- {email}" for email in found_emails)
    else:
        return f"No emails found matching query: {query}"


def read_email(email_id: str) -> str:
    """Read a specific email by ID.
    
    Args:
        email_id: The ID of the email to read
        
    Returns:
        The email content
    """
    # Mock email database
    mock_email_content = {
        "1": "Subject: Quarterly Report\nFrom: John <john@company.com>\n\nHi team, please find attached the Q4 quarterly report. The numbers look good overall with 15% growth.",
        "2": "Subject: Follow-up on Q4 numbers\nFrom: John <john@company.com>\n\nJust wanted to follow up on yesterday's meeting about the Q4 numbers. Any questions?",
        "3": "Subject: Budget discussions for next quarter\nFrom: Finance Team\n\nWe need to finalize the budget for Q1. Please review the attached spreadsheet.",
        "4": "Subject: Budget approval needed\nFrom: Finance Team\n\nThe budget proposal requires your approval before we can proceed.",
        "5": "Subject: Latest updates from Sarah\nFrom: Sarah <sarah@company.com>\n\nHere are this week's project updates and key milestones achieved.",
        "6": "Subject: Sarah's weekly summary\nFrom: Sarah <sarah@company.com>\n\nWeekly summary of completed tasks and upcoming deadlines."
    }
    
    if email_id in mock_email_content:
        return f"Email {email_id} content:\n\n{mock_email_content[email_id]}"
    else:
        return f"Email with ID '{email_id}' not found."