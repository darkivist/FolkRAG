import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """1""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2024""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Smart Web Scraping Data Generator""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to develop an advanced, intelligent web scraping ecosystem that revolutionizes 
            the way data is extracted from the internet. This system will:

            1. Incorporate both classical and cutting-edge web scraping techniques, including but not limited to:
               - BeautifulSoup and Scrapy for traditional HTML parsing
               - Selenium and Playwright for browser automation
               - Puppeteer for headless Chrome control
               - AJAX and API interaction capabilities
               - Scrapy-Splash for JavaScript rendering
               - Newspaper3k for article extraction
               - Octoparse for visual scraping
               - ParseHub for complex scraping scenarios

            2. Automatically analyze given domains to identify potential data sources across the internet.

            3. Intelligently select the most appropriate scraping method and tool for each identified source, 
            considering factors such as website structure, dynamic content, and anti-scraping measures.

            4. Implement a confidence scoring system for text extraction, assessing the reliability and accuracy of the
             scraped data.

            5. Provide robust data cleaning, sorting, and structuring functionalities to ensure high-quality output.

            6. Offer a user-friendly interface for configuring scraping tasks and visualizing results.

            The end product will be a versatile, self-optimizing toolbox that can adapt to various web scraping 
            scenarios, significantly reducing manual intervention and improving the efficiency and accuracy of data 
            collection processes.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The dataset for this project will be dynamically generated through web scraping. Students will identify and 
            target multiple websites across different domains to ensure a diverse range of data structures and 
            challenges. Potential data sources may include:

            1. E-commerce websites for product information
            2. News portals for article content and metadata
            3. Social media platforms for public posts and user information
            4. Government websites for public data and statistics
            5. Job posting sites for employment trends and requirements

            The exact websites and data types will be determined during the project's initial phase, ensuring compliance
             with each site's terms of service and ethical scraping practices.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            In the age of big data, efficient and intelligent web scraping tools are crucial for gathering large-scale,
             real-time information from the internet. This project addresses several key needs:

            1. Automation of data collection processes, saving time and resources for researchers and businesses
            2. Improvement of data quality through advanced cleaning and sorting techniques
            3. Adaptation to various web structures and anti-scraping measures using both classical and modern approaches
            4. Creation of a versatile toolbox that can be applied across different industries and research fields
            5. Enhancing students' skills in web technologies, data processing, and software development

            By developing this smart web scraping toolbox, students will contribute to the field of data acquisition and
             processing, potentially benefiting various sectors including market research, academic studies, and 
             data-driven decision-making in businesses.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will be approached through several key steps:

            1. Research and Analysis:
               - Study existing web scraping techniques and tools
               - Analyze challenges in current web scraping practices
               - Identify potential target websites and their structures

            2. Design and Development of Scraping Modules:
               - Implement classical scraping techniques (e.g., using BeautifulSoup, Scrapy)
               - Develop modules for modern techniques (e.g., headless browsers, API interactions)
               - Create a system to handle dynamic content and JavaScript-rendered pages

            3. Intelligent Tool Selection System:
               - Develop algorithms to analyze website structures and content
               - Create a decision-making system to choose the best scraping tool for each scenario
               - Implement machine learning models to improve tool selection over time

            4. Confidence Scoring Mechanism:
               - Design and implement a scoring system for assessing text extraction accuracy
               - Develop methods to validate and cross-reference extracted data
               - Create visualizations for confidence scores

            5. Data Cleaning and Processing:
               - Develop robust data cleaning algorithms to handle inconsistencies and errors
               - Implement intelligent data parsing to extract structured information from unstructured content
               - Create sorting and categorization functionalities based on various criteria

            6. Integration and Automation:
               - Combine all modules into a cohesive ecosystem
               - Develop a user-friendly interface for configuring and running scraping tasks
               - Implement scheduling and monitoring features for automated data collection

            7. Testing and Optimization:
               - Conduct thorough testing on various websites and data types
               - Optimize performance and handle rate limiting and other restrictions
               - Ensure scalability for large-scale data collection tasks

            8. Documentation and Deployment:
               - Create comprehensive documentation for the toolbox
               - Prepare a deployment strategy for easy installation and use
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project:

            - (2 Weeks) Research and Planning
            - (4 Weeks) Development of Core Scraping Modules
            - (3 Weeks) Intelligent Tool Selection System
            - (2 Weeks) Confidence Scoring Mechanism
            - (3 Weeks) Data Cleaning and Processing Implementation
            - (3 Weeks) Integration and Automation
            - (2 Weeks) Testing and Optimization
            - (1 Week)  Documentation
            - (1 Week)  Final Presentation and Project Wrap-up
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is suitable for a team of 3-4 students. The complexity and scope of the project allow for 
            effective distribution of tasks among team members, promoting collaborative learning and development.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Several challenges may arise during the project:

            1. Ethical and Legal Considerations: Ensuring compliance with websites' terms of service and respecting robots.txt files.
            2. Anti-Scraping Measures: Dealing with CAPTCHAs, IP blocking, and other protective measures implemented by websites.
            3. Handling Dynamic Content: Effectively scraping websites with heavy JavaScript usage and dynamic loading.
            4. Data Quality and Consistency: Ensuring the scraped data is accurate, complete, and properly structured across different sources.
            5. Performance and Scalability: Optimizing the toolbox to handle large-scale scraping tasks efficiently.
            6. Keeping Up with Web Technologies: Adapting to new web technologies and scraping techniques as they emerge.
            7. Cross-Platform Compatibility: Ensuring the toolbox works across different operating systems and environments.
            8. Accuracy of Tool Selection: Developing a reliable system for choosing the best scraping method for each scenario.
            9. Confidence Scoring Complexity: Creating an accurate and meaningful confidence scoring system for extracted data.
            10. Integration Challenges: Seamlessly combining various scraping tools and techniques into a unified system.

            Students will need to research and implement solutions to these challenges, which will be an integral part of the learning experience.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Amir Jafari",
        "Proposed by email": "ajafari@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/amir-jafari/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
