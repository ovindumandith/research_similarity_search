import spacy
import json

nlp = spacy.load('en_core_web_lg')
descriptions = [
"Elastic Scaling - The capability of cloud systems to adjust computing resources in real-time based on usage, improving both performance and cost management",  
"Cloud Service Model - A cloud model that delivers computing resources and services over the internet, allowing users to access infrastructure, platforms, and software without owning them",  
"Virtual Cloud Infrastructure - The provisioning of virtualized computing resources, including storage and processing power, to streamline IT management in a cloud environment",  
"On-Demand Cloud Resources - A model that allows users to access cloud resources as needed, without long-term commitments, optimizing both cost and efficiency",  
"Cloud Scalability - The cloud's capacity to scale resources automatically, ensuring optimal performance during peak demand periods and cost savings when usage drops",  
"Public Cloud - A cloud computing environment where resources are shared and owned by a third-party provider, offering services to multiple customers on a pay-as-you-go basis",  
"Private Cloud - A cloud infrastructure that is used exclusively by a single organization, providing enhanced control and security over its resources",  
"Cloud Data Storage - A model for storing data in the cloud, making it accessible remotely and protected by the provider’s backup and redundancy systems",  
"Multi-cloud Strategy - Using multiple cloud services from different providers to avoid vendor lock-in and increase resilience across platforms",  
"Cloud Computing Security - Techniques and practices aimed at ensuring the safety of data and applications stored and processed in cloud environments",  
"API Integration in Cloud - The process of linking cloud-based systems with other applications and services using APIs, enabling seamless communication across platforms",  
"Cloud Data Centers - Facilities used by cloud service providers to house servers and storage systems that deliver cloud computing resources to end-users",  
"Cloud Hosting - The practice of hosting websites, applications, or services on remote servers, managed and maintained by a third-party provider",  
"Cloud Backup Solutions - Cloud-based services designed to automatically back up data from local systems, ensuring disaster recovery and protection from data loss",  
"Cloud Migration - The process of transferring applications, data, or other business elements from on-premise systems to cloud environments",  
"Cloud Monitoring - The continuous tracking of cloud-based services, applications, and infrastructure to ensure they function efficiently and securely",  
"Cloud Automation - The use of software to automate the management and provisioning of cloud resources, reducing manual intervention and optimizing resource usage",  
"Cost Optimization in Cloud - The process of identifying and implementing strategies to reduce unnecessary cloud service spending while maintaining required performance",  
"Disaster Recovery in Cloud - Cloud-based solutions designed to recover data and applications following system failures or catastrophic events",  
"Cloud Elastic Load Balancing - A mechanism that distributes incoming traffic across multiple cloud resources to ensure high availability and performance",  
"Virtual Cloud Servers - Virtualized computing resources that mimic dedicated servers in the cloud, offering flexibility, scalability, and cost efficiency",  
"Cloud-based Collaboration Tools - Software hosted in the cloud that enables team collaboration, communication, and document sharing from anywhere",  
"Cloud Security Compliance - Adherence to regulatory and security standards by cloud providers to ensure data protection and privacy in the cloud environment",  
"Cloud Disaster Recovery as a Service (DRaaS) - A service that allows businesses to back up and recover data using cloud-based resources after a disaster",  
"Cloud Resource Management - The process of planning, deploying, and optimizing cloud resources to meet business needs while minimizing costs",  
"Cloud Computing for Big Data - The use of cloud computing resources to process, analyze, and store large volumes of data at scale",  
"Cloud-native Infrastructure - IT infrastructure specifically designed for cloud environments, leveraging technologies like containers and microservices for flexibility",  
"Cloud Virtual Machines - Virtualized computing environments in the cloud that can run applications and workloads independently of physical hardware",  
"Cloud Application Delivery - The process of delivering software applications via the cloud, ensuring scalability, reliability, and global accessibility",  
"Cloud Containerization - A method of packaging software and its dependencies together in containers, ensuring consistency across environments in the cloud",  
"Serverless Architecture - A cloud computing model where developers focus on code without the need to manage underlying server infrastructure",  
"Cloud-enabled Data Analytics - The use of cloud computing to run data analytics tools and services, enabling insights from large datasets",  
"Cloud Resource Provisioning - The allocation of cloud computing resources, such as storage and processing power, as required by the user's needs",  
"Cloud SaaS Platform - A software delivery model where applications are hosted on a provider’s servers and delivered to users over the internet",  
"Infrastructure Management in the Cloud - Managing and configuring cloud-based IT resources like storage, servers, and networking to ensure smooth operations",  
"Cloud Networking - The use of virtualized networks in the cloud to connect cloud resources with each other and with end-user devices securely",  
"Virtual Cloud Environments - Environments created through virtualization technologies that simulate dedicated physical servers within a cloud setting",  
"Cloud-based Virtual Desktops - Cloud services that allow users to access desktop computing environments remotely, with all data and applications hosted in the cloud",  
"Cloud-native DevOps - The use of DevOps practices in cloud environments to improve application development, deployment, and monitoring",  
"Cloud Service Level Agreement (SLA) - A formal agreement between a cloud provider and a customer detailing the performance, availability, and support levels expected",  
"Cloud Multi-tenancy - A cloud architecture where a single instance of an application or service serves multiple customers or tenants securely",  
"Cloud Virtual Private Network (VPN) - A secure, encrypted connection from a user’s device to a cloud environment, ensuring privacy and protection of sensitive data",  
"Cloud Load Balancer - A service that automatically distributes incoming network traffic across multiple cloud resources to ensure high availability and responsiveness",  
"Cloud-based Monitoring Tools - Tools designed to track the performance, uptime, and health of cloud resources and applications, often with real-time alerts",  
"Cloud Data Encryption - The process of encrypting data stored or transmitted in cloud environments to protect it from unauthorized access or theft",  
"Cloud Data Warehousing - Storing large amounts of data in the cloud in a structured manner, allowing for complex analytics and reporting",  
"Private Cloud Hosting - A type of hosting where cloud resources are dedicated to a single organization for greater security and control",  
"Cloud Compliance Standards - The set of laws and regulations that cloud providers must follow to ensure security, privacy, and data protection",  
"Cloud Resource Allocation - The distribution of computing, storage, and networking resources across users and workloads in the cloud to ensure optimal performance",  
"Cloud-based Application Scaling - The ability to increase or decrease cloud resources allocated to an application based on its traffic or load",  
"Cloud Service Management - The process of managing and maintaining cloud services to ensure they operate efficiently and meet business needs",  
"Public Cloud Security - Security measures and protocols implemented by cloud providers to protect resources and data in a shared cloud environment",  
"Cloud API Gateway - A tool that provides a single entry point for managing and securing API calls in cloud-based services",  
"Cloud Application Monitoring - Tracking and analyzing the performance of applications deployed in the cloud to ensure they meet desired performance metrics",  
"Hybrid Cloud Architecture - A blend of on-premise data centers and public or private cloud environments, allowing for flexibility and optimized resource usage",  
"Cloud DevOps Tools - Software tools used in cloud environments to automate development and operations workflows, speeding up product delivery",  
"Cloud Infrastructure Automation - Automating the deployment and management of cloud resources to reduce the need for manual intervention",  
"Virtual Private Cloud (VPC) - A private network hosted within a public cloud, providing secure, isolated resources that can be controlled and configured",  
"Cloud-native Security Tools - Security solutions designed for cloud environments that integrate with the infrastructure to provide real-time protection",  
"Cloud-based Disaster Recovery - Leveraging cloud resources to replicate critical systems and data, enabling quick recovery during unexpected events",  
"Public Cloud Infrastructure - Cloud resources that are publicly available and managed by third-party providers for shared access by different customers",  
"Cloud Data Migration - The process of transferring data from on-premise systems to cloud storage solutions for better scalability and accessibility",  
"Cloud Computing for IoT - The use of cloud infrastructure to support the storage, processing, and management of Internet of Things (IoT) devices and data",  
"Cloud Application Deployment - The process of uploading and configuring an application to run in a cloud environment for global access",  
"Cloud Hosting for eCommerce - Using cloud resources to host eCommerce websites, ensuring scalability, reliability, and performance during peak traffic periods",  
"Cloud Performance Tuning - Optimizing cloud infrastructure and applications to ensure they perform efficiently and meet user requirements",  
"Cloud Cost Management - Strategies and tools used to track, monitor, and optimize cloud expenses, ensuring cost-effective use of resources",  
"Cloud Service Orchestration - The coordination and automation of cloud services and resources to create complex workflows or business processes",  
"Containerized Cloud Applications - Software applications packaged in containers to ensure portability and scalability across cloud environments",  
"Cloud Backup as a Service (BaaS) - A cloud service that automatically backs up user data to the cloud, ensuring protection against data loss",  
"Cloud-based Identity Management - Systems that manage identity and access for users across cloud environments, ensuring secure access control",
"Disaster Recovery in Cloud - Cloud-based solutions designed to recover data and applications following system failures or catastrophic events"
    

]
# Define target vector dimension (128)
VECTOR_DIMENSION = 128

# Function to pad or truncate vectors
def pad_or_truncate_vector(vector, target_dim):
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))

# Generate embeddings and create JSON entries
description_vectors_list = []
for description in descriptions:
    truncated_description = description[:250]  # Truncate to 250 characters
    doc = nlp(truncated_description)  # Get spaCy vector
    vector = pad_or_truncate_vector(doc.vector.tolist(), VECTOR_DIMENSION)  # Adjust to 128 dimensions

    # Create entry matching Milvus schema
    entry = {
        "vector": vector,
        "description": truncated_description,
    }
    description_vectors_list.append(entry)

# Save to JSON
with open("descriptions_vectors.json", "w") as json_file:
    json.dump(description_vectors_list, json_file, indent=2)

print("Embedding data saved to descriptions_vectors.json")