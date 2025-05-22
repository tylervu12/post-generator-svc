import argparse
import json
from datetime import datetime
from .post_generator import PostGenerator
from .models import PostPromptInput

def main():
    parser = argparse.ArgumentParser(description="Generate social media posts using AI")
    parser.add_argument("prompt", help="The prompt for generating the post")
    parser.add_argument("--platform", choices=["twitter", "linkedin"], required=True,
                      help="The platform to generate the post for")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Create input data
    input_data = PostPromptInput(
        prompt=args.prompt,
        platform=args.platform
    )
    
    try:
        # Initialize generator
        generator = PostGenerator()
        
        # Generate post
        result = generator.generate_post(input_data)
        
        # Convert to dict for JSON serialization
        result_dict = result.model_dump()
        
        # Add timestamp
        result_dict["timestamp"] = datetime.now().isoformat()
        
        # Output result
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_dict, f, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print("\nInput Context:")
            print("-" * 50)
            print(f"Prompt: {args.prompt}")
            print(f"Platform: {args.platform}")
            
            print("\nRetrieved Context (Example Posts):")
            print("-" * 50)
            for i, ctx in enumerate(result.generated_post.retrieved_contexts, 1):
                print(f"\nExample {i}:")
                print(f"Content: {ctx.content}")
                print(f"Platform: {ctx.platform}")
                print(f"Date: {ctx.date}")
                print(f"Similarity Score: {ctx.similarity_score:.2f}")
            
            print("\nGenerated Post:")
            print("-" * 50)
            print(result.generated_post.post)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 