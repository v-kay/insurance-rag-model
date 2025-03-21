import boto3
import logging
from botocore.exceptions import ClientError
import json

class BedrockWrapper:
  
    def __init__(self,service,region):
        
        """ Initiates the bedrock client and runtime
        """
        self.bedrock_client = boto3.client(service_name=service, region_name=region)
        self.bedrock_runtime = boto3.client('bedrock-runtime')

    def list_foundation_models(self):
        """ List the foundational models available
        """
        response = self.bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        print(f"Got {len(models)} foundation models.", models)


    def set_model(self,model_id):
        """ sets the generative AI models Id to be used
        """
        self.model_id=model_id


    def generate_body(self,prompt,params):

        """ sets model parameter and prompt
        """

        body=json.dumps({
             'prompt': prompt,
             **params
        })

        return body

    def invoke_model(self,body):
        """calls the model and get response string
        """
      
        accept = 'application/json'
        contentType = 'application/json'
        # Define one or more messages using the "user" and "assistant" roles.
        message_list = [{"role": "user", "content": [{"text": body}]}]
        request_body = {
    "schemaVersion": "messages-v1",
    "messages": message_list,
    #"system": system_list,
    #"inferenceConfig": inf_params,
}

        # response = self.bedrock_runtime.invoke_model(body=body, modelId= self.model_id, 
        #                                              accept=accept, contentType=contentType)
        # response = self.bedrock_runtime.invoke_model(messages=message_list, modelId= self.model_id, 
        #                                              accept=accept, contentType=contentType)
        # system = [{ "text": "You are a helpful assistant" }]

        messages = [
            {"role": "user", "content": [{"text": "Write a short story about dragons"}]},
        ]

        inf_params = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}

        additionalModelRequestFields = {
            "inferenceConfig": {
                "topK": 20
            }
        }

        model_response = self.bedrock_runtime.converse(
            modelId=self.model_id, 
            messages=message_list, 
            # system=system, 
            # inferenceConfig=inf_params,
            # additionalModelRequestFields=additionalModelRequestFields
        )

        # print("\n[Full Response]")
        # print(json.dumps(model_response, indent=2))

        # print("\n[Response Content Text]")
        # print(model_response["output"]["message"]["content"][0]["text"])
        result = model_response["output"]["message"]["content"][0]["text"]
        return result



if __name__ == "__main__":

    bedrock=BedrockWrapper("bedrock","us-east-1")

    #bedrock.list_foundation_models()

    #modelId = 'anthropic.claude-v2'
    modelId = 'amazon.nova-micro-v1:0'

    bedrock.set_model(modelId)

    params={
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.1,
    }

    transcript_example_text=f'''
    James: Hi Sarah, shall we go over the status of the Maxwell account marketing project?
    Sarah: Sure, that sounds good. What's on your agenda for today?
    James: I wanted to touch base on the timeline we discussed last week and see if we're
    still on track to hit those deadlines. How did the focus group go earlier this week?
    Sarah: The focus group went pretty well overall. We got some good feedback on the new
    branding concepts that I think will help refine our ideas. The one hiccup is that the
    product samples we were hoping to show arrived late, so we weren't able to do the
    unboxing and product trial portion that we had planned.
    James: Okay, good to hear it was mostly positive. Sorry to hear about the product
    sample delay. Did that impact our ability to get feedback on the premium packaging
    designs?
    Sarah: It did a little bit - we weren't able to get as detailed feedback on unboxing
    experience and the tactile elements we were hoping for. But we did get high-level
    input at least. I'm hoping to schedule a follow up focus group in two weeks when the
    samples arrive.
    James: Sounds good. Please keep me posted on when that follow up will happen. On the
    plus side, it's good we built in a buffer for delays like this. As long as we can get
    the second round of feedback by mid-month, I think we can stay on track.Sarah: Yes, 
    I'll make sure to get that second session booked and keep you in the loop.
    How are things looking for the website development and SEO optimization? Still on pace
    for the planned launch?
    James: We're in good shape there. The initial site map and wireframes are complete and
    we began development work this week. I'm feeling confident we can hit our launch goal
    of March 15th if all goes smoothly from here on out. One request though - can you send
    me any new branding assets or guidelines as soon as possible? That will help ensure it
    gets incorporated properly into the site design.
    Sarah: Sure, will do. I should have those new brand guidelines over to you by early
    next week once we finalize with Maxwell.
    James: Sounds perfect, thanks! Let's plan to meet again next Thursday and review the
    focus group results and new launch timeline in more detail.
    Sarah: Works for me! I'll get those calendar invites out.
    James: Great, talk to you then.
    Sarah: Thanks James, bye! 
    '''
   

    prompt=f'''
    Human: I am going to give you transcript of a meeting extract key informations, 
    members of meeting and minutes of the meeting,key takeaways and next step . 
    Here is the transcript:<transcript>{transcript_example_text}</transcript> 
    Assistant:'''
    
    body=bedrock.generate_body(prompt,params)

    result=bedrock.invoke_model(body)

    print(result)
